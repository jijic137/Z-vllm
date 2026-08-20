"""KV int8 vs bf16 的 decode-PPL 对拍（teacher-forced 增量 decode，走真实分页 KV cache 路径）。

为什么不用 w8_ppl.py 的独立 forward：那条路径 block_tables=None、k_cache 为空，
prefill 直接用新算出的全精度 K/V，根本不读 KV cache，因此测不到 KV 量化。
KV 量化的影响只体现在 decode 步的分页读（triton_paged_decode 读 int8 cache）。

本脚本对固定语料做 teacher-forced 增量 decode：
  对每条序列，逐步 (t=0..L-1) 处理 token t：把 token t 的 KV 写入分页 cache，
  以 context_lens=t+1 读历史（含自身，标准 causal self-term）算出位置 t 的 hidden，
  用 lm_head 投影成 logits 预测 token t+1，累计交叉熵。
  - kv16：cache 为 bf16，存的就是新 K/V 同值 -> decode-PPL 应≈非分页 prefill-PPL（自校验）
  - kv8：cache 为 int8 per-token 量化 -> decode-PPL 反映真实量化影响
  ΔPPL = PPL(kv8) - PPL(kv16)，并输出 kv16 decode 与 prefill 的差作为脚本正确性自检。

用法：HIP_VISIBLE_DEVICES=2 ~/zvllm-env/bin/python kv_ppl.py <model_dir>
"""
import math
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from zvllm.models import build_model
from zvllm.utils.loader import load_model
from zvllm.utils.context import set_context

BLOCK_SIZE = 256

# 与 w8_ppl.py 共用同一语料（保证可比），英文 + 中文混合，约 1-2k token
CORPUS = [
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that learn from data. A neural network is composed of "
    "layers of interconnected nodes, each of which applies a weighted sum "
    "followed by a non-linear activation function.",
    "In distributed training, gradient aggregation is a critical bottleneck. "
    "Data parallelism replicates the model across devices, while model "
    "parallelism splits the model into pieces. Tensor parallelism partitions "
    "individual layers, and pipeline parallelism assigns different layers to "
    "different stages of a pipeline.",
    "推理引擎的核心在于高效地管理键值缓存。分页注意力将键值缓存切分为固定 "
    "大小的块，通过块表把逻辑位置映射到物理位置，从而减少显存碎片并支持前缀 "
    "共享。调度器在每一步决定哪些序列进入预填充阶段，哪些进入解码阶段。",
    "The history of computing is a history of shrinking. Transistors have "
    "halved in size for decades, following a pattern that many thought would "
    "continue indefinitely. Each generation brought faster clocks, larger "
    "memory, and lower cost per operation, enabling entirely new classes of "
    "applications that were previously infeasible.",
    "大语言模型通过自注意力机制捕捉长距离依赖。旋转位置编码将位置信息编码为 "
    "旋转矩阵，使得注意力分数只依赖于相对位置。归一化层稳定了训练过程，使得 "
    "深层网络的梯度既不消失也不爆炸。",
    "A compiler translates source code into an efficient machine representation. "
    "The optimizer applies a sequence of transformations, some local and some "
    "global, to reduce the cost of the generated code. Register allocation is "
    "one of the hardest problems, because it must decide which values to keep "
    "in fast registers and which to spill to memory.",
]


def attach_kv_cache(model, hf_config, kv_bits, max_blocks):
    """分配分页 KV cache（kv_bits 16/8）并挂到每个 Attention 模块（与 model_runner 约定一致）。"""
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    L = hf_config.num_hidden_layers
    kv_int8 = (kv_bits == 8)
    cache_dtype = torch.int8 if kv_int8 else hf_config.dtype
    kv_cache = torch.zeros(2, L, max_blocks, BLOCK_SIZE, num_kv_heads, head_dim,
                           dtype=cache_dtype, device="cuda")
    kv_scale = None
    if kv_int8:
        kv_scale = torch.zeros(2, L, max_blocks, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    layer_id = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]
            if kv_int8:
                module.k_scale = kv_scale[0, layer_id]
                module.v_scale = kv_scale[1, layer_id]
            layer_id += 1
    return kv_cache, kv_scale


def prefill_ppl(model, hf_config, seqs):
    """非分页 prefill PPL（等价 w8_ppl，block_tables=None，全精度）。用作 kv16 decode 的自检基准。"""
    total_ce, total_n = 0.0, 0
    with torch.no_grad():
        for ids in seqs:
            L = len(ids)
            input_ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
            positions = torch.arange(L, dtype=torch.int64, device="cuda")
            cu = torch.tensor([0, L], dtype=torch.int32, device="cuda")
            set_context(is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=L, max_seqlen_k=L, slot_mapping=None, block_tables=None)
            hidden = model(input_ids, positions)
            logits = F.linear(hidden, model.lm_head.weight)
            logits = logits[:-1].float()
            labels = input_ids[1:]
            total_ce += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        labels.reshape(-1), reduction="sum").item()
            total_n += L - 1
    return math.exp(total_ce / total_n), total_n


def decode_ppl(model, hf_config, seqs, kv_bits):
    """teacher-forced 增量 decode PPL，走分页 KV cache（kv_bits 16/8）。"""
    max_L = max(len(s) for s in seqs)
    max_blocks = (max_L + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_cache, kv_scale = attach_kv_cache(model, hf_config, kv_bits, max_blocks)
    dev = "cuda"
    total_ce, total_n = 0.0, 0
    with torch.no_grad():
        for ids in seqs:
            L = len(ids)
            nblk = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
            # 本序列只用块 0..nblk-1；清零避免跨序列残留
            kv_cache[:, :, :nblk].zero_()
            if kv_scale is not None:
                kv_scale[:, :, :nblk].zero_()
            block_table = torch.zeros(1, nblk, dtype=torch.int32, device=dev)
            for i in range(nblk):
                block_table[0, i] = i
            for t in range(L):
                input_ids = torch.tensor([ids[t]], dtype=torch.int64, device=dev)
                positions = torch.tensor([t], dtype=torch.int64, device=dev)
                context_lens = torch.tensor([t + 1], dtype=torch.int32, device=dev)
                slot_mapping = torch.tensor([t], dtype=torch.int32, device=dev)    # 块 0 起连续 -> slot=t
                set_context(is_prefill=False, context_lens=context_lens,
                            block_tables=block_table, slot_mapping=slot_mapping)
                hidden = model(input_ids, positions)
                if t < L - 1:
                    logits = F.linear(hidden, model.lm_head.weight).float()
                    total_ce += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                 torch.tensor([ids[t + 1]], dtype=torch.int64, device=dev),
                                                 reduction="sum").item()
                    total_n += 1
    del kv_cache
    if kv_scale is not None:
        del kv_scale
    torch.cuda.empty_cache()
    return math.exp(total_ce / total_n), total_n


def main():
    model_dir = sys.argv[1]
    torch.manual_seed(0)
    hf_config = AutoConfig.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    seqs = [tok.encode(text) for text in CORPUS]
    seqs = [s for s in seqs if 0 < len(s) <= 1024]

    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29513",
                                world_size=1, rank=0)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")
    model = build_model(hf_config, SimpleNamespace(weight_bits=16))
    load_model(model, model_dir)
    model.eval()
    torch.set_default_dtype(old_dtype)
    torch.set_default_device("cpu")

    # 非分页 prefill PPL（全精度基准）
    ppl_pre, n_pre = prefill_ppl(model, hf_config, seqs)
    print(f"[PPL] tokens={n_pre}  prefill(非分页,全精度) PPL = {ppl_pre:.4f}", flush=True)
    # 分页 decode PPL：kv16（自校验）与 kv8（量化影响）
    ppl_d16, n_d16 = decode_ppl(model, hf_config, seqs, 16)
    print(f"[PPL] tokens={n_d16}  decode-KV-bf16 PPL = {ppl_d16:.4f}   "
          f"(自检: 与 prefill 差 {ppl_d16 - ppl_pre:+.4f}, 应很小)", flush=True)
    ppl_d8, n_d8 = decode_ppl(model, hf_config, seqs, 8)
    print(f"[PPL] tokens={n_d8}  decode-KV-int8 PPL = {ppl_d8:.4f}", flush=True)
    delta = ppl_d8 - ppl_d16
    print(f"[PPL] ΔPPL(int8 vs bf16) = {delta:+.4f}  (验收线 <= +0.1)", flush=True)
    print(f"[PPL] {'PASS' if delta <= 0.1 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
