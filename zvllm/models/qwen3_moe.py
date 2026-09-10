import torch
from torch import nn
import torch.distributed as dist

from zvllm.layers.activation import SiluAndMul
from zvllm.layers.fused_moe import BMM_MAX_TOKENS, FUSED_MAX_TOKENS, fused_moe_bmm, fused_moe_triton, triton_available
from zvllm.layers.layernorm import RMSNorm
from zvllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from zvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from zvllm.models.qwen3 import Qwen3Attention, Qwen3MLP


def is_moe_layer(config, layer_id: int) -> bool:
    """判定第 layer_id 层用稀疏 MoE 还是稠密 MLP（与 HF Qwen3MoeDecoderLayer 语义一致）。

    HF 原式：
        (layer_id not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0)
    即 mlp_only_layers 里的层恒为稠密层；其余层按 decoder_sparse_step（每 N 层一个 MoE 层，
    step=1 即全部）取模决定。Qwen3-30B-A3B / 235B-A22B 均为 mlp_only_layers=[] + step=1
    （全层 MoE），而部分衍生 checkpoint（如 PrimeIntellect/qwen3-moe-tiny）用
    mlp_only_layers=[0] 把第 0 层设成稠密层。

    另外兼容 qwen2_moe 家族的 first_k_dense_replace（Qwen3-MoE config 不含该字段，
    getattr 兜底为 0，因此对 Qwen3-MoE 无影响）。
    """
    if layer_id in (getattr(config, "mlp_only_layers", None) or []):
        return False
    if layer_id < (getattr(config, "first_k_dense_replace", 0) or 0):
        return False
    num_experts = getattr(config, "num_experts", 0) or 0
    sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
    return num_experts > 0 and (layer_id + 1) % sparse_step == 0


class Qwen3MoeExpert(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        tp_group: "dist.ProcessGroup | None",
        quantized: bool = False,
    ) -> None:
        super().__init__()
        # 单个专家：与 dense MLP 同构，但 linear 按"专家内 TP"组切分
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [moe_intermediate_size] * 2,
            bias=False,
            tp_group=tp_group,
            quantized=quantized,
        )
        self.down_proj = RowParallelLinear(
            moe_intermediate_size,
            hidden_size,
            bias=False,
            tp_group=tp_group,
            quantized=quantized,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.gate_up_proj(x))
        return self.down_proj(x)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    Router + 专家。支持两个正交的并行维度（乘积等于全局 TP）：
    - 专家内 TP（moe_tp）：每个专家的 linear 在组内 rank 间切分，每 rank 累加
      全部本地专家的 partial 结果，层内一次组内 all_reduce 即完成专家计算；
    - 专家并行（moe_ep）：专家按组均分，每 rank 只算自己拥有的专家。
      "对专家加权和"是线性运算，全 world 一次 all_reduce 与 all-to-all 等价
      （省掉数据搬运，代价是通信量为 token 数的 topk 倍）。
    小批次（T <= FUSED_MAX_TOKENS）走 fused_moe 融合路径（grouped GEMM，零 host 同步）；
    大批次保留遍历式 forward（可读优先）：每专家一个 gather + GEMM + scatter 累加。
    """

    def __init__(
        self,
        config,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = config.num_experts_per_tok
        self.tp_size = moe_tp_size
        self.ep_size = moe_ep_size
        self.tp_group = tp_group
        # 本 rank 属于 EP 组 (rank // moe_tp)，拥有连续一段全局专家号
        ep_rank = dist.get_rank() // moe_tp_size
        num_local = self.num_experts // moe_ep_size
        self.local_expert_ids = list(range(ep_rank * num_local, (ep_rank + 1) * num_local))
        # ModuleDict 保留全局专家号进参数名（mlp.experts.{e}.*），与 checkpoint 命名对齐
        self.experts = nn.ModuleDict({
            str(e): Qwen3MoeExpert(config.hidden_size, config.moe_intermediate_size, tp_group,
                                   quantized=quantized)
            for e in self.local_expert_ids
        })
        # router 全量复制在每个 rank：输入跨 rank 一致，top-k 结果天然一致，无需通信
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.local_start = self.local_expert_ids[0]
        # stacked 专家权重：w13 [E, 2I, H]（前 I 行 gate、后 I 行 up），w2 [E, H, I]。
        # 大 buffer 是权重的唯一存储：每个专家的 linear 权重把 param.data 重指向 buffer
        # 切片（view），loader 的 copy_ 经 view 直接写入大 buffer。融合路径（grouped GEMM）
        # 直接用 buffer：零拷贝、零额外显存——若首次 forward 才惰性 torch.cat，会完整复制
        # 一份专家权重（30B EP=2 约 +29GB/rank，必 OOM，2026-08-19 真机实测）；遍历路径
        # 则继续用 per-expert view 读同一份权重。
        first = self.experts[str(self.local_expert_ids[0])]
        gw, dw = first.gate_up_proj.weight, first.down_proj.weight
        num_local = len(self.local_expert_ids)
        self.w13 = torch.empty(num_local, *gw.shape, dtype=gw.dtype, device=gw.device)
        self.w2 = torch.empty(num_local, *dw.shape, dtype=dw.dtype, device=dw.device)
        if quantized:
            # W8：scale 大 buffer 与权重 buffer 走同样的 view 重指向机制，
            # loader 经 expert 参数 view 把量化结果写进大 buffer
            gs, ds = first.gate_up_proj.weight_scale, first.down_proj.weight_scale
            self.w13_scale = torch.empty(num_local, *gs.shape, dtype=gs.dtype, device=gs.device)
            self.w2_scale = torch.empty(num_local, *ds.shape, dtype=ds.dtype, device=ds.device)
        else:
            self.w13_scale = self.w2_scale = None
        if quantized and torch.cuda.is_available() and not triton_available():
            # 量化 MoE 生产路径必须 kernel 内反量化（bmm 整层物化会让专家权重
            # 显存翻倍）；GPU 上无可用 Triton 的量化模型启动即报错
            raise RuntimeError("W8 量化 MoE 需要可用 Triton（当前 CUDA 环境无）")
        for i, e in enumerate(self.local_expert_ids):
            expert = self.experts[str(e)]
            expert.gate_up_proj.weight.data = self.w13[i]
            expert.down_proj.weight.data = self.w2[i]
            if quantized:
                expert.gate_up_proj.weight_scale.data = self.w13_scale[i]
                expert.down_proj.weight_scale.data = self.w2_scale[i]

    def _forward_loop(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """遍历式实现：每专家一个 gather + GEMM + scatter 累加（含 host 同步，大批次 GEMM 足够大时无所谓）"""
        out = torch.zeros(x.shape[0], self.topk, x.shape[1], device=x.device, dtype=x.dtype)
        for e_id, expert in self.experts.items():
            mask = topk_ids == int(e_id)
            if not mask.any():
                continue
            idx = mask.nonzero()
            y = expert(x[idx[:, 0]])
            w = topk_weights[idx[:, 0], idx[:, 1]].to(x.dtype).unsqueeze(-1)
            out[idx[:, 0], idx[:, 1]] = y * w
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 路由（softmax → top-k → 归一化，与 Qwen3-MoE 参考实现一致）
        router_probs = torch.softmax(self.gate(x).float(), dim=-1)
        topk_weights, topk_ids = torch.topk(router_probs, self.topk, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        # 输出按 (token, slot) 组织，非本地专家的位置为 0，跨 rank 求和即 all_reduce
        T = x.shape[0]
        use_triton = T <= FUSED_MAX_TOKENS and x.is_cuda and triton_available()
        use_bmm = T <= BMM_MAX_TOKENS and not use_triton
        if use_triton or use_bmm:
            w13, w2 = self.w13, self.w2
            out = (fused_moe_triton if use_triton else fused_moe_bmm)(
                x, topk_weights, topk_ids, self.local_start, w13, w2,
                self.w13_scale, self.w2_scale
            )
        else:
            out = self._forward_loop(x, topk_weights, topk_ids)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        if self.ep_size > 1:
            dist.all_reduce(out)
        return out.sum(dim=1)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_id: int,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            quantized=quantized,
        )
        if is_moe_layer(config, layer_id):
            self.mlp = Qwen3MoeSparseMoeBlock(config, moe_tp_size, moe_ep_size, tp_group,
                                               quantized=quantized)
        else:
            # 稠密层用 config.intermediate_size（与 HF Qwen3MoeMLP 一致，注意不是 moe_intermediate_size）
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quantized=quantized,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
        quantized: bool = False,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, i, moe_tp_size, moe_ep_size, tp_group, quantized)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    has_moe = True

    def __init__(
        self,
        config,
        moe_tp_size: int,
        moe_ep_size: int,
        tp_group: "dist.ProcessGroup | None",
        quantized: bool = False,
    ) -> None:
        super().__init__()
        # EP > 1 时 checkpoint 里有不属于本 rank 的专家权重，loader 需要跳过
        self.skip_unowned_weights = moe_ep_size > 1
        self.model = Qwen3MoeModel(config, moe_tp_size, moe_ep_size, tp_group, quantized)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
