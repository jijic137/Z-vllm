"""MoE 稠密层选择纯 CPU 单测（无需 GPU）。运行：python tests/test_moe_layer_select.py

背景：HF Qwen3MoeDecoderLayer 判定某层是稀疏 MoE 还是稠密 MLP 的规则是

    (layer_id not in config.mlp_only_layers) and (
        config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0)

本仓库早期只认 qwen2_moe 家族的 first_k_dense_replace，导致带 mlp_only_layers 的
Qwen3-MoE checkpoint（如 PrimeIntellect/qwen3-moe-tiny，mlp_only_layers=[0]）第 0 层
被建成稀疏块，加载时报 "Qwen3MoeSparseMoeBlock has no attribute `down_proj`"。
本文件固化判定规则与层结构命名，防止回归。
"""
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from transformers import Qwen3MoeConfig  # noqa: E402

from zvllm.models.qwen3_moe import (  # noqa: E402
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
    is_moe_layer,
)
from zvllm.models.qwen3 import Qwen3MLP  # noqa: E402


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


def test_official_qwen3_moe_configs():
    """Qwen3-30B-A3B / 235B-A22B 的官方字段：全部层都是 MoE 层。"""
    print("test_official_qwen3_moe_configs")
    qwen3_30b = Qwen3MoeConfig(num_hidden_layers=48, num_experts=128, num_experts_per_tok=8,
                               decoder_sparse_step=1, mlp_only_layers=[])
    qwen3_235b = Qwen3MoeConfig(num_hidden_layers=94, num_experts=128, num_experts_per_tok=8,
                                decoder_sparse_step=1, mlp_only_layers=[])
    for cfg, n in ((qwen3_30b, 48), (qwen3_235b, 94)):
        assert all(is_moe_layer(cfg, i) for i in range(n)), "官方 Qwen3-MoE 应当全层稀疏"
    # 缺省值：Qwen3MoeConfig 的 mlp_only_layers 默认 None（HF 内部转 []）、step 默认 1
    default_cfg = Qwen3MoeConfig(num_hidden_layers=4)
    assert default_cfg.mlp_only_layers in (None, []), default_cfg.mlp_only_layers
    assert default_cfg.decoder_sparse_step == 1
    print("  ok")


def test_mlp_only_layers_dense_head():
    """PrimeIntellect/qwen3-moe-tiny 形态：第 0 层稠密，其余层 MoE。"""
    print("test_mlp_only_layers_dense_head")
    cfg = SimpleNamespace(num_experts=16, num_experts_per_tok=4, decoder_sparse_step=1,
                          mlp_only_layers=[0], num_hidden_layers=24)
    flags = [is_moe_layer(cfg, i) for i in range(5)]
    assert flags == [False, True, True, True, True], flags
    print("  ok")


def test_decoder_sparse_step():
    """decoder_sparse_step=N：每 N 层留一个 MoE 层（(i+1) % N == 0 的层才是 MoE）。"""
    print("test_decoder_sparse_step")
    cfg = SimpleNamespace(num_experts=60, num_experts_per_tok=4, decoder_sparse_step=2,
                          mlp_only_layers=[], num_hidden_layers=6)
    flags = [is_moe_layer(cfg, i) for i in range(6)]
    assert flags == [False, True, False, True, False, True], flags
    # mlp_only_layers 与 sparse_step 叠加时，显式列出的层优先级更高
    cfg2 = SimpleNamespace(num_experts=60, num_experts_per_tok=4, decoder_sparse_step=2,
                           mlp_only_layers=[3], num_hidden_layers=6)
    assert [is_moe_layer(cfg2, i) for i in range(6)] == [False, True, False, False, False, True]
    print("  ok")


def test_first_k_dense_replace_compat():
    """兼容字段 first_k_dense_replace（qwen2_moe 家族）：前 k 层稠密。"""
    print("test_first_k_dense_replace_compat")
    cfg = SimpleNamespace(num_experts=8, decoder_sparse_step=1, mlp_only_layers=[],
                          first_k_dense_replace=2, num_hidden_layers=4)
    assert [is_moe_layer(cfg, i) for i in range(4)] == [False, False, True, True]
    print("  ok")


def test_edge_cases():
    """无专家 / 缺字段 / 字段为 None：一律退化成稠密层，不抛异常。"""
    print("test_edge_cases")
    assert not any(is_moe_layer(SimpleNamespace(num_experts=0), i) for i in range(3))
    assert not is_moe_layer(SimpleNamespace(), 0), "缺 num_experts 字段时按 HF 语义应为稠密层"
    cfg = SimpleNamespace(num_experts=8, decoder_sparse_step=None, mlp_only_layers=None)
    assert all(is_moe_layer(cfg, i) for i in range(3))
    print("  ok")


def test_layer_structure():
    """真实建模：稠密层出 Qwen3MLP（gate_up_proj 命名），MoE 层出稀疏块（含 experts）。"""
    print("test_layer_structure")
    cfg = Qwen3MoeConfig(vocab_size=32, hidden_size=64, intermediate_size=128,
                         moe_intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
                         mlp_only_layers=[0], max_position_embeddings=64)
    model = Qwen3MoeForCausalLM(cfg, moe_tp_size=1, moe_ep_size=1, tp_group=None)
    layers = model.model.layers
    assert isinstance(layers[0].mlp, Qwen3MLP), "mlp_only_layers=[0] 的第 0 层应为稠密 MLP"
    assert isinstance(layers[1].mlp, Qwen3MoeSparseMoeBlock)
    assert isinstance(layers[2].mlp, Qwen3MoeSparseMoeBlock)
    names = dict(model.named_parameters())
    # 参数名与 checkpoint 键对齐：稠密层是 model.layers.0.mlp.gate_proj/up_proj/down_proj
    # （经 packed_modules_mapping 映射到 zvllm 的 gate_up_proj）
    assert "model.layers.0.mlp.gate_up_proj.weight" in names
    assert "model.layers.0.mlp.down_proj.weight" in names
    # MoE 层：专家参数保留全局专家号，稀疏块无 down_proj 直连属性（回归点）
    assert "model.layers.1.mlp.experts.0.gate_up_proj.weight" in names
    assert "model.layers.1.mlp.gate.weight" in names
    assert not hasattr(layers[1].mlp, "down_proj")
    assert not hasattr(layers[0].mlp, "experts")
    print("  ok")


if __name__ == "__main__":
    test_official_qwen3_moe_configs()
    test_mlp_only_layers_dense_head()
    test_decoder_sparse_step()
    test_first_k_dense_replace_compat()
    test_edge_cases()
    test_layer_structure()
    print("ALL PASSED")
