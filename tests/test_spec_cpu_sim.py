"""投机解码 CPU 全链路仿真（确定性目标模型；spec on/off 必须逐位一致）。运行：python tests/test_spec_cpu_sim.py

与 test_spec_decode.py（调度器级单测）互补：本文件用真实模型 + 真实 ModelRunner
（prepare_prefill/decode -> forward -> LM head 行过滤 -> 采样/接受-拒绝）在完整引擎
步循环里跑，只把"目标分布"换成确定性函数 T(input_token, position)——权重仍是真实
forward（随机权重，embed/attention/KV 缓存/RMSNorm 全走真实代码路径）。

为什么两次运行的官方 token 流必须一致（正确性基准）：
- spec off：x_j = T(x_{j-1}, j-1)（每步回喂最后一个 token）
- spec on：验证步第 i 行（位置 L-1+i）argmax = T(input_i, L-1+i)；首行输入是
  官方末 token，必等于 x_L；某草稿被接受后，下一行输入正是该 token，也必等于
  目标；首次拒绝取目标 argmax 并停止。-> 每步产出的官方 token 与 spec-off
  贪心逐位相同。流若分歧必是记账/状态 bug（调度/接受回写/块/哈希），与权重无关。

T 取周期函数，使 n-gram 草稿高频命中，强制大量验证步、多 token 回写、块跨越与抢占。
覆盖：
1) 纯位置周期目标（几乎每步全接受）x 3 序列并发（chunked prefill + 混合批 / 整批 prefill）
2) 依赖 token 的目标（接受/拒绝混合）x 3 序列
3) 单序列（隔离批内交互）
4) 收紧 KV 块数强制抢占 + prefix cache 复用（spec on/off 都要一致）
"""
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
try:
    torch._dynamo.config.disable = True    # CPU 单测环境跳过 torch.compile（Windows 无 C++ 编译器）
except AttributeError:
    pass
import torch.distributed as dist

from transformers import LlamaConfig

import zvllm.layers.attention as attn_mod
from zvllm.engine.model_runner import ModelRunner
from zvllm.engine.scheduler import Scheduler
from zvllm.engine.sequence import Sequence
from zvllm.layers.sampler import Sampler
from zvllm.models.llama import LlamaForCausalLM
from zvllm.sampling_params import SamplingParams

# ---------- 小模型几何（全部 CPU 可跑） ----------
VOCAB, HIDDEN = 32, 64
N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM = 2, 4, 2, 16
INTER, MAX_POS = 128, 512
BS = 4                      # 小块：每产出 5 token 跨 1+ 块，块记账高频受压
Sequence.block_size = BS
EOS = 31                    # 词表内但目标函数永不产出
CYC = [10, 11, 12, 13, 14, 15, 16, 17]   # 8 周期：保证 4-gram 周期复现 -> n-gram 草稿高频命中


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def init_single_rank():
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_free_port()}",
                                world_size=1, rank=0)


init_single_rank()


# ---------- CPU 替身（ModelRunner 无条件 .pin_memory().cuda()；triton kernel 不可用） ----------
def _store_kvcache_cpu(key, value, k_cache, v_cache, slot_mapping):
    """store_kvcache_kernel 的 CPU 等价实现（语义相同：按 slot_mapping 逐行写分页缓存）。"""
    N, H, D = key.shape
    kf = k_cache.view(k_cache.size(0) * k_cache.size(1), H * D)
    vf = v_cache.view(v_cache.size(0) * v_cache.size(1), H * D)
    m = slot_mapping >= 0
    if m.any():
        idx = slot_mapping[m].long()
        kf[idx] = key[m].view(-1, H * D)
        vf[idx] = value[m].view(-1, H * D)


@contextmanager
def cpu_shims():
    saved = (torch.Tensor.cuda, torch.Tensor.pin_memory, attn_mod.store_kvcache, torch.tensor)
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    attn_mod.store_kvcache = _store_kvcache_cpu
    # torch.tensor(..., pin_memory=True) 是工厂关键字参数（CPU 无 pin allocator）：剥掉
    _real_tensor = torch.tensor
    def _tensor_no_pin(*a, **k):
        k.pop("pin_memory", None)
        return _real_tensor(*a, **k)
    torch.tensor = _tensor_no_pin
    try:
        yield
    finally:
        torch.Tensor.cuda, torch.Tensor.pin_memory, attn_mod.store_kvcache, torch.tensor = saved


# ---------- 确定性目标模型 ----------
def make_target(mode):
    """T(token, pos) -> 预测位置 pos+1 的 token。
    cycle：只看位置，x_j = CYC[j % 8] -> 4-gram 每 8 位复现，草稿恒为真延续（全接受）；
    twist：加入 token 依赖，流不再是固定周期（接受/拒绝混合），但状态 (pos%8, token)
           转移是 64 态排列，流仍纯周期（周期为 8 的倍数），4-gram 照样复现。"""
    table = torch.zeros(VOCAB, MAX_POS, dtype=torch.int64)
    for t in range(VOCAB):
        for p in range(MAX_POS):
            idx = (p + 1 + t) % 8 if mode == "twist" else (p + 1) % 8
            table[t, p] = CYC[idx]
    return table


class TargetModel:
    """真实 LlamaForCausalLM + 目标函数包装：body 走真实 forward（attention/KV/RMSNorm），
    只在进入真实 ParallelLMHead 前对 hidden 的目标维加主导偏移 -> 最终 logit argmax ==
    T(input_token, position)，而 LM head 行过滤（spec_flags 保留规则）仍在链路中受测。"""

    def __init__(self, table):
        cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
                          num_attention_heads=N_HEADS, num_key_value_heads=N_KV_HEADS,
                          intermediate_size=INTER, max_position_embeddings=MAX_POS,
                          hidden_act="silu", rope_theta=10000, tie_word_embeddings=False)
        self.model = LlamaForCausalLM(cfg)
        g = torch.Generator().manual_seed(2024)
        with torch.no_grad():
            for p in self.model.parameters():
                p.normal_(0.0, 0.05, generator=g)
        # lm_head 置为恒等投影：logit_j = hidden_j（j < VOCAB），argmax 完全由 hidden 决定
        w = self.model.lm_head.weight.data
        w.zero_()
        w[torch.arange(VOCAB), torch.arange(VOCAB)] = 1.0
        self.table = table
        orig = self.model.model.forward

        def body(input_ids, positions):
            h = orig(input_ids, positions)
            h = h.clone()
            h[:, :VOCAB] -= 100.0
            tgt = self.table[input_ids, positions]
            h[torch.arange(len(input_ids)), tgt] += 200.0
            return h

        self.model.model.forward = body

    def attach_kv(self, num_blocks):
        kv = torch.zeros(2, N_LAYERS, num_blocks, BS, N_KV_HEADS, HEAD_DIM)
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.attn.k_cache = kv[0, i]
            layer.self_attn.attn.v_cache = kv[1, i]


class CpuRunner(ModelRunner):
    """继承 ModelRunner 以复用 prepare_prefill/prepare_decode/_sample_all 等真实实现；
    绕过 __init__（它要拉 NCCL/CUDA），手动装配 CPU 可用的最小状态。"""

    def __init__(self, target_model, cfg):
        # 不调用 super().__init__
        self.config = cfg
        self.model = target_model.model
        self.sampler = Sampler()
        self.block_size = BS
        self.enforce_eager = True
        self.world_size = 1
        self.rank = 0


def run_engine(spec, target_mode, prompts, num_blocks, budget, max_tokens=96):
    """跑完整引擎循环，返回 (各序列官方补全 token 流, spec_stats, 抢占次数)。"""
    cfg = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=budget, eos=EOS,
                          kvcache_block_size=BS, num_kvcache_blocks=num_blocks,
                          max_model_len=MAX_POS, spec_decode=spec, spec_gamma=4, spec_ngram=4)
    tm = TargetModel(make_target(target_mode))
    tm.attach_kv(num_blocks)
    runner = CpuRunner(tm, cfg)
    sched = Scheduler(cfg)
    state = {"preempts": 0}
    orig_preempt = sched.preempt

    def counting_preempt(seq):
        state["preempts"] += 1
        return orig_preempt(seq)

    sched.preempt = counting_preempt
    all_seqs = []
    for prompt in prompts:
        s = Sequence(prompt, SamplingParams(temperature=0, max_tokens=max_tokens))
        sched.add(s)
        all_seqs.append(s)
    with cpu_shims():
        while not sched.is_finished():
            seqs, is_prefill = sched.schedule()
            token_lists = runner.run(seqs, is_prefill)
            sched.postprocess_multi(seqs, token_lists, is_prefill)
    streams = [s.completion_token_ids for s in all_seqs]
    return streams, dict(sched.spec_stats), state["preempts"]


def _diff(off, on):
    for i, (a, b) in enumerate(zip(off, on)):
        for j in range(max(len(a), len(b))):
            ta, tb = a[j] if j < len(a) else None, b[j] if j < len(b) else None
            if ta != tb:
                return f"seq{i} pos{j}: off={ta} on={tb} (len off={len(a)} on={len(b)})"
    return f"长度不一致 off={[len(a) for a in off]} on={[len(b) for b in on]}"


PROMPTS3 = [list(range(10)), [5, 6, 7, 8], [2, 3, 4, 5, 6]]


def test_multi_seq_cycle_target():
    # budget=16：chunked prefill + prefill/verify 混合批；budget=64：整批 prefill
    for budget in (16, 64):
        off, _, _ = run_engine("off", "cycle", PROMPTS3, 256, budget)
        on, stats, _ = run_engine("ngram", "cycle", PROMPTS3, 256, budget)
        assert on == off, (budget, _diff(off, on))
        assert [len(s) for s in on] == [96] * 3, [len(s) for s in on]
        assert stats["steps"] >= 40 and stats["accepted_tokens"] >= 150, stats
    print(f"  ok 3 序列并发 x 周期目标（budget 16/64）：逐位一致，"
          f"steps/accepted 达标（on 运行统计 {stats}）")


def test_multi_seq_twist_target():
    off, _, _ = run_engine("off", "twist", PROMPTS3, 256, 16)
    on, stats, _ = run_engine("ngram", "twist", PROMPTS3, 256, 16)
    assert on == off, _diff(off, on)
    assert [len(s) for s in on] == [96] * 3
    # twist 流的周期长于 8：接受率低于 cycle，但仍须有实质验证活动
    assert stats["steps"] >= 30 and stats["accepted_tokens"] >= 40, stats
    print(f"  ok 3 序列 x token 依赖目标（混合接受/拒绝）：逐位一致（统计 {stats}）")


def test_single_seq_cycle_target():
    # 单序列：隔离批内交互（GPU 症状是 3 序列批内分歧）
    prompt = list(range(8))
    off, _, _ = run_engine("off", "cycle", [prompt], 256, 16)
    on, stats, _ = run_engine("ngram", "cycle", [prompt], 256, 16)
    assert on == off, _diff(off, on)
    assert len(on[0]) == 96
    assert stats["steps"] >= 15 and stats["accepted_tokens"] >= 60, stats
    print(f"  ok 单序列 x 周期目标：逐位一致（统计 {stats}）")


def test_tight_blocks_preemption():
    # 块数收紧到"单条序列装得下（75 token -> 19 块 <= 24）、三条同时装不下
    # （3 x 19 = 57 > 24）"：必发抢占；被抢占的投机序列 re-prefill 走 prefix cache 复用
    results = {}
    for spec in ("off", "ngram"):
        streams, stats, preempts = run_engine(spec, "cycle", PROMPTS3, 24, 16, max_tokens=64)
        assert preempts >= 1, f"{spec}: 未触发抢占（测试无效），preempts={preempts}"
        results[spec] = (streams, preempts)
    off, p_off = results["off"]
    on, p_on = results["ngram"]
    assert on == off, _diff(off, on)
    assert [len(s) for s in on] == [64] * 3
    print(f"  ok 紧块（24 块）强制抢占 + prefix 复用：on/off 逐位一致"
          f"（preempts off={p_off}, on={p_on}）")


if __name__ == "__main__":
    test_multi_seq_cycle_target()
    test_multi_seq_twist_target()
    test_single_seq_cycle_target()
    test_tight_blocks_preemption()
    print("ALL SPEC CPU SIM TESTS PASSED")
