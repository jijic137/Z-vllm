"""LLaMA/Qwen2 模型家族支持纯 CPU 单测（无需 GPU/triton/flash_attn）。运行：python tests/test_models_llama.py

对拍对象：
- config 解析：config.json（llama: norm_eps / qwen2: rms_norm_eps）-> AutoConfig -> Config 早期校验，
  含 transformers 5.x 兼容点（rope_theta 并入 rope_scaling/rope_parameters dict；
  norm_eps 不并入 rms_norm_eps，后者保留默认值）
- 权重加载：HF 标准命名（model.layers.N.self_attn.q_proj ...）-> packed qkv / gate_up 布局
- 全模型 forward：prefill logits 对拍 naive 参考实现（RMSNorm + RoPE + GQA + SwiGLU，float32）
- model_type 分发：build_model 返回正确类，不支持的家族报清晰错误
"""
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

try:
    torch._dynamo.config.disable = True    # 单测环境跳过 torch.compile，直接 eager
except AttributeError:
    pass
import torch.distributed as dist
from safetensors.torch import save_file

from transformers import AutoConfig, LlamaConfig, Qwen2Config, Qwen3Config

from zvllm.config import Config
from zvllm.models import build_model, SUPPORTED_MODEL_TYPES
from zvllm.models.llama import LlamaForCausalLM
from zvllm.models.qwen3 import Qwen3ForCausalLM
from zvllm.utils.context import set_context
from zvllm.utils.loader import load_model

# ---------- 小模型几何（全部 CPU 可跑） ----------
VOCAB, HIDDEN, N_LAYERS = 32, 64, 2
N_HEADS, N_KV_HEADS, HEAD_DIM = 4, 2, 16
INTER, MAX_POS = 128, 64
T_SEQ = 6    # 单条 prefill 序列的 token 数


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def init_single_rank():
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{free_port()}", world_size=1, rank=0)


init_single_rank()
torch.manual_seed(0)


def check_close(name, got, expect, atol):
    got, expect = got.float(), expect.float()
    diff = (got - expect).abs().max().item()
    assert diff <= atol, f"{name}: max diff {diff} > {atol}"


# ---------- 参考实现（naive，float32） ----------

def ref_rmsnorm(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w


def ref_rmsnorm_add(x, res, w, eps):
    xf = x.float() + res.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w, xf


def ref_rope(x, pos, theta):
    D = x.size(-1)
    inv = theta ** (-torch.arange(0, D, 2, dtype=torch.float) / D)
    ang = pos[:, None].float() * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def ref_gqa_attention(q, k, v, scale):
    T, H = q.size(0), q.size(1)
    r = H // k.size(1)
    k = k.repeat_interleave(r, dim=1)
    v = v.repeat_interleave(r, dim=1)
    s = torch.einsum("lhd,mhd->hlm", q, k) * scale
    mask = ~torch.tril(torch.ones(T, T, dtype=torch.bool))
    s = s.masked_fill(mask.unsqueeze(0), float("-inf"))
    return torch.einsum("hlm,mhd->lhd", s.softmax(-1), v)


def ref_forward(sd, input_ids, positions, eps, theta):
    """naive 完整 forward，返回每个位置的 logits [T, vocab]"""
    T = input_ids.size(0)
    h = sd["model.embed_tokens.weight"][input_ids]
    res = None
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        if res is None:
            res = h
            h = ref_rmsnorm(h, sd[p + "input_layernorm.weight"], eps)
        else:
            h, res = ref_rmsnorm_add(h, res, sd[p + "input_layernorm.weight"], eps)
        q = (h @ sd[p + "self_attn.q_proj.weight"].T).view(T, N_HEADS, HEAD_DIM)
        k = (h @ sd[p + "self_attn.k_proj.weight"].T).view(T, N_KV_HEADS, HEAD_DIM)
        v = (h @ sd[p + "self_attn.v_proj.weight"].T).view(T, N_KV_HEADS, HEAD_DIM)
        q, k = ref_rope(q, positions, theta), ref_rope(k, positions, theta)
        o = ref_gqa_attention(q, k, v, HEAD_DIM ** -0.5).reshape(T, -1)
        h = o @ sd[p + "self_attn.o_proj.weight"].T
        h, res = ref_rmsnorm_add(h, res, sd[p + "post_attention_layernorm.weight"], eps)
        gate = h @ sd[p + "mlp.gate_proj.weight"].T
        up = h @ sd[p + "mlp.up_proj.weight"].T
        h = torch.nn.functional.silu(gate) * up
        h = h @ sd[p + "mlp.down_proj.weight"].T
    h, _ = ref_rmsnorm_add(h, res, sd["model.norm.weight"], eps)
    return h @ sd["lm_head.weight"].T


# ---------- 测试夹具 ----------

def write_cfg_json(dirpath, model_type, eps_field, eps, theta):
    dirpath.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": model_type,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV_HEADS,
        "intermediate_size": INTER,
        "max_position_embeddings": MAX_POS,
        "hidden_act": "silu",
        "rope_theta": theta,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        eps_field: eps,
    }
    (dirpath / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return dirpath


LLAMA_EPS, LLAMA_THETA = 1e-5, 500000
QWEN2_EPS, QWEN2_THETA = 1e-6, 1000000


def load_hf_config(base, model_type, eps_field, eps, theta):
    """走真实路径：config.json -> AutoConfig（与线上加载一致）"""
    d = write_cfg_json(base / model_type, model_type, eps_field, eps, theta)
    return AutoConfig.from_pretrained(str(d)), d


def make_hf_state_dict(seed):
    """按 HF 标准命名生成一组随机权重（模拟真实 checkpoint 的 key 布局）"""
    g = torch.Generator().manual_seed(seed)
    t = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=g)}
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        t[p + "self_attn.q_proj.weight"] = torch.randn(N_HEADS * HEAD_DIM, HIDDEN, generator=g)
        t[p + "self_attn.k_proj.weight"] = torch.randn(N_KV_HEADS * HEAD_DIM, HIDDEN, generator=g)
        t[p + "self_attn.v_proj.weight"] = torch.randn(N_KV_HEADS * HEAD_DIM, HIDDEN, generator=g)
        t[p + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, N_HEADS * HEAD_DIM, generator=g)
        t[p + "mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN, generator=g)
        t[p + "mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN, generator=g)
        t[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER, generator=g)
        # norm 权重偏离 1，确保 *weight 这一步真的生效
        t[p + "input_layernorm.weight"] = 1 + 0.01 * torch.randn(HIDDEN, generator=g)
        t[p + "post_attention_layernorm.weight"] = 1 + 0.01 * torch.randn(HIDDEN, generator=g)
    t["model.norm.weight"] = 1 + 0.01 * torch.randn(HIDDEN, generator=g)
    t["lm_head.weight"] = torch.randn(VOCAB, HIDDEN, generator=g)
    return t


# ---------- 测试 ----------

def test_config_llama(base):
    print("test_config_llama")
    cfg, d = load_hf_config(base, "llama", "norm_eps", LLAMA_EPS, LLAMA_THETA)
    engine_cfg = Config(model=str(d))
    assert isinstance(engine_cfg.hf_config, LlamaConfig)
    assert engine_cfg.max_model_len == MAX_POS    # 默认 4096 应被夹到模型 max_position_embeddings
    model = LlamaForCausalLM(engine_cfg.hf_config)
    # norm_eps（而非 transformers 5.x 里保留默认值的 rms_norm_eps）流进 norm 层
    assert model.model.layers[0].input_layernorm.eps == LLAMA_EPS
    assert model.model.norm.eps == LLAMA_EPS
    # rope_theta 经 rope_scaling/rope_parameters dict 生效
    cache = model.model.layers[0].self_attn.rotary_emb.cos_sin_cache
    inv = LLAMA_THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM)
    ang = torch.tensor([7.0])[:, None] * inv[None, :]
    check_close("llama rope cos", cache[7, 0, :HEAD_DIM // 2], ang.cos()[0], atol=1e-5)
    check_close("llama rope sin", cache[7, 0, HEAD_DIM // 2:], ang.sin()[0], atol=1e-5)
    print("  ok")


def test_config_qwen2(base):
    print("test_config_qwen2")
    cfg, d = load_hf_config(base, "qwen2", "rms_norm_eps", QWEN2_EPS, QWEN2_THETA)
    engine_cfg = Config(model=str(d))
    assert isinstance(engine_cfg.hf_config, Qwen2Config)
    model = LlamaForCausalLM(engine_cfg.hf_config)
    assert model.model.layers[0].input_layernorm.eps == QWEN2_EPS
    cache = model.model.layers[0].self_attn.rotary_emb.cos_sin_cache
    inv = QWEN2_THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM)
    ang = torch.tensor([7.0])[:, None] * inv[None, :]
    check_close("qwen2 rope cos", cache[7, 0, :HEAD_DIM // 2], ang.cos()[0], atol=1e-5)
    print("  ok")


def test_config_unsupported(base):
    print("test_config_unsupported")
    d = write_cfg_json(base / "gpt2", "gpt2", "norm_eps", 1e-5, 10000)
    try:
        Config(model=str(d))
        raise AssertionError("应当拒绝不支持的 model_type")
    except AssertionError as e:
        assert "model_type" in str(e), str(e)
    print("  ok")


def test_weight_loading(base):
    print("test_weight_loading")
    sd = make_hf_state_dict(0)
    wdir = base / "weights_llama"
    wdir.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(wdir / "model.safetensors"))
    cfg, _ = load_hf_config(base, "llama", "norm_eps", LLAMA_EPS, LLAMA_THETA)
    model = LlamaForCausalLM(cfg)
    load_model(model, str(wdir))
    w = model.model.layers[0].self_attn.qkv_proj.weight.data
    check_close("qkv.q", w[0:64], sd["model.layers.0.self_attn.q_proj.weight"], atol=0)
    check_close("qkv.k", w[64:96], sd["model.layers.0.self_attn.k_proj.weight"], atol=0)
    check_close("qkv.v", w[96:128], sd["model.layers.0.self_attn.v_proj.weight"], atol=0)
    gu = model.model.layers[0].mlp.gate_up_proj.weight.data
    check_close("gate_up.gate", gu[0:128], sd["model.layers.0.mlp.gate_proj.weight"], atol=0)
    check_close("gate_up.up", gu[128:256], sd["model.layers.0.mlp.up_proj.weight"], atol=0)
    check_close("embed", model.model.embed_tokens.weight.data, sd["model.embed_tokens.weight"], atol=0)
    check_close("lm_head", model.lm_head.weight.data, sd["lm_head.weight"], atol=0)
    check_close("final_norm", model.model.norm.weight.data, sd["model.norm.weight"], atol=0)
    print("  ok")


def test_attention_bias_inference(base):
    """config 缺 attention_bias 字段（Qwen2-0.5B 发行版）：引擎 Config 从 checkpoint 推断，
    有 bias 则建参数并精确加载，无 bias 则不建参数。"""
    print("test_attention_bias_inference")
    for has_bias in (True, False):
        sd = make_hf_state_dict(3)
        if has_bias:
            for i in range(N_LAYERS):
                sd[f"model.layers.{i}.self_attn.q_proj.bias"] = torch.randn(N_HEADS * HEAD_DIM, generator=torch.Generator().manual_seed(3 + i))
                sd[f"model.layers.{i}.self_attn.k_proj.bias"] = torch.randn(N_KV_HEADS * HEAD_DIM, generator=torch.Generator().manual_seed(7 + i))
                sd[f"model.layers.{i}.self_attn.v_proj.bias"] = torch.randn(N_KV_HEADS * HEAD_DIM, generator=torch.Generator().manual_seed(11 + i))
        mdir = base / f"model_bias_{has_bias}"
        write_cfg_json(mdir, "qwen2", "rms_norm_eps", QWEN2_EPS, QWEN2_THETA)
        save_file(sd, str(mdir / "model.safetensors"))
        # 模拟该发行版：config 对象上删掉 attention_bias 字段（部分 transformers 版本默认就有）
        hf = AutoConfig.from_pretrained(str(mdir))
        if hasattr(hf, "attention_bias"):
            del hf.attention_bias
        engine_cfg = Config(model=str(mdir))
        assert engine_cfg.hf_config.attention_bias is has_bias, \
            f"has_bias={has_bias} 时推断应为 {has_bias}，实际 {engine_cfg.hf_config.attention_bias}"
        model = LlamaForCausalLM(engine_cfg.hf_config)
        load_model(model, str(mdir))
        qw, kw = N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
        for i in range(N_LAYERS):
            b = model.model.layers[i].self_attn.qkv_proj.bias
            if not has_bias:
                assert b is None, "无 bias checkpoint 不应创建 bias 参数"
                continue
            b = b.data
            check_close(f"qkv.bias.q L{i}", b[0:qw], sd[f"model.layers.{i}.self_attn.q_proj.bias"], atol=0)
            check_close(f"qkv.bias.k L{i}", b[qw:qw + kw], sd[f"model.layers.{i}.self_attn.k_proj.bias"], atol=0)
            check_close(f"qkv.bias.v L{i}", b[qw + kw:qw + 2 * kw], sd[f"model.layers.{i}.self_attn.v_proj.bias"], atol=0)
    print("  ok")


def test_forward(base, model_type, eps_field, eps, theta, seed):
    print(f"test_forward_{model_type}")
    sd = make_hf_state_dict(seed)
    wdir = base / f"weights_{model_type}"
    wdir.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(wdir / "model.safetensors"))
    cfg, _ = load_hf_config(base, model_type, eps_field, eps, theta)
    model = LlamaForCausalLM(cfg)
    load_model(model, str(wdir))
    input_ids = torch.randint(0, VOCAB, (T_SEQ,))
    positions = torch.arange(T_SEQ)
    # 单序列 prefill、无 prefix（block_tables=None 路径）；k_cache 未分配 -> 不写缓存
    set_context(True, cu_seqlens_q=torch.tensor([0, T_SEQ], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, T_SEQ], dtype=torch.int32),
                max_seqlen_q=T_SEQ, max_seqlen_k=T_SEQ,
                slot_mapping=torch.zeros(T_SEQ, dtype=torch.int32))
    with torch.inference_mode():
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)
    ref = ref_forward(sd, input_ids, positions, eps, theta)
    assert hidden.shape == (T_SEQ, HIDDEN), hidden.shape
    assert logits.shape == (1, VOCAB), logits.shape    # prefill 只返回每条序列最后一个 token
    check_close("final logits", logits[0], ref[T_SEQ - 1], atol=2e-4)
    print("  ok (max diff %.2e)" % (logits[0] - ref[T_SEQ - 1]).abs().max().item())


def test_tied_embeddings():
    print("test_tied_embeddings")
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
                      num_attention_heads=N_HEADS, num_key_value_heads=N_KV_HEADS,
                      intermediate_size=INTER, max_position_embeddings=MAX_POS,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    # torch 2.7：.data 访问器每次返回新包装对象，is 不可靠；用 data_ptr + 变异确认共享存储
    assert model.lm_head.weight.data.data_ptr() == model.model.embed_tokens.weight.data.data_ptr()
    model.model.embed_tokens.weight.data[0, 0] = 123.0
    assert model.lm_head.weight.data[0, 0].item() == 123.0
    print("  ok")


def test_build_model_dispatch(base):
    print("test_build_model_dispatch")
    llama_cfg, _ = load_hf_config(base, "llama", "norm_eps", LLAMA_EPS, LLAMA_THETA)
    q2_cfg, _ = load_hf_config(base, "qwen2", "rms_norm_eps", QWEN2_EPS, QWEN2_THETA)
    assert isinstance(build_model(llama_cfg, None), LlamaForCausalLM)
    assert isinstance(build_model(q2_cfg, None), LlamaForCausalLM)
    d = write_cfg_json(base / "qwen3", "qwen3", "rms_norm_eps", QWEN2_EPS, QWEN2_THETA)
    q3_cfg = AutoConfig.from_pretrained(str(d))
    assert isinstance(build_model(q3_cfg, None), Qwen3ForCausalLM)
    assert set(SUPPORTED_MODEL_TYPES) >= {"qwen3", "qwen3_moe", "llama", "qwen2"}

    class FakeCfg:
        model_type = "deepseek_v3"

    try:
        build_model(FakeCfg(), None)
        raise AssertionError("应当拒绝不支持的 model_type")
    except ValueError as e:
        assert "deepseek_v3" in str(e), str(e)
    print("  ok")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        test_config_llama(base)
        test_config_qwen2(base)
        test_config_unsupported(base)
        test_weight_loading(base)
        test_attention_bias_inference(base)
        test_forward(base, "llama", "norm_eps", LLAMA_EPS, LLAMA_THETA, seed=1)
        test_forward(base, "qwen2", "rms_norm_eps", QWEN2_EPS, QWEN2_THETA, seed=2)
        test_tied_embeddings()
        test_build_model_dispatch(base)
    print("ALL PASSED")
