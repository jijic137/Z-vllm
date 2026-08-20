<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Z-vLLM

基于开源项目二次开发实现的轻量级 LLM 推理引擎。

完整实现现代 LLM 推理的核心机制：PagedAttention、continuous batching、prefix caching、chunked prefill、张量并行、专家并行（MoE）、CUDA graph、torch.compile。推理吞吐与 vLLM 相当（见 [Benchmark](#benchmark)）。

## 特性

* 🚀 **快速离线推理** — 吞吐与 vLLM 相当
* 📖 **可读的代码库** — 约 1,900 行 Python，完整推理管线清晰可见
* 🖥️ **CUDA / ROCm 双平台** — 同一套代码支持 NVIDIA 与 AMD GPU，flash-attn 缺失时自动 SDPA 兜底
* 📦 **PagedAttention** — KV cache 按 256-token 块管理，显存无碎片
* 🔁 **Continuous batching** — 以迭代为粒度动态拼批
* ✂️ **Chunked prefill** — 长 prompt 分块 prefill，与 decode 交错执行
* 🧩 **Prefix caching** — 相同前缀跨请求共享，节省 prefill 计算
* 🧮 **张量并行** — 支持 1–8 卡
* 🧱 **专家并行（EP）** — MoE 专家切分到多卡，专家内 TP × EP 任意组合；decode 阶段 Triton grouped-GEMM 融合路径
* 🪞 **DP（多副本）推理** — 进程级 D×TP 副本（各自独立 KV 池与调度器），父进程 round-robin 路由 + 同步 step 驱动，per-replica 看门狗 + 全局 fail-fast；`api_server --dp-size`
* 🗜️ **权重量化（W8A16）** — `weight_bits=8` 加载时 int8 量化（per-group 128 对称），kernel 内反量化融合 GEMM；Qwen3-30B-A3B 57GB → 约 29GB，单张 48GB 卡跑 30B MoE
* 🪃 **投机解码（n-gram 草稿）** — 零训练自历史 n-gram 草稿，验证步单次 forward 接受-拒绝，输出分布与目标严格一致；贪心 A/B 一致性门槛 + 性能基准
* ⚡ **CUDA graph + torch.compile** — 捕获 decode 图，降低 launch 开销
* 🤖 **支持 Qwen3 / LLaMA / Qwen2 模型家族（稠密 + MoE）** — 如 Qwen3-0.6B、Qwen3-30B-A3B、Llama-3.2-1B、Qwen2-0.5B
* 🎲 **完整采样** — 贪心（temperature=0）、top-k、top-p、seed 可复现
* 📡 **流式输出** — `generate(..., stream=True)` 逐 token 产出事件
* 🔌 **OpenAI 兼容服务** — FastAPI + SSE，`/v1/chat/completions`、`/v1/completions` 直接对接 OpenAI SDK
* 🛑 **stop 与逐请求取消** — `stop` 停止串直通；流式请求客户端断连自动取消并释放 KV 块；GPU 阶段看门狗防服务 wedged

## 项目结构

```
.
├── example.py                # 快速上手示例
├── example_qwen3_moe.py      # Qwen3-30B-A3B MoE 示例（TP / EP）
├── bench.py                  # 吞吐基准测试脚本
├── tests/                    # 纯 CPU 单元测试（调度 / 块管理 / 流式 / API 服务）
├── assets/logo.png
└── zvllm/
    ├── __init__.py           # 对外 API：LLM、SamplingParams
    ├── llm.py                # LLM 入口（即 LLMEngine）
    ├── sampling_params.py    # 采样参数
    ├── config.py             # 引擎配置
    ├── engine/
    │   ├── llm_engine.py     # 主引擎：请求接入、调度循环、输出汇总
    │   ├── scheduler.py      # continuous batching + chunked prefill 调度
    │   ├── block_manager.py  # PagedAttention 块分配 + prefix cache
    │   ├── model_runner.py   # 每 rank 的模型执行（CUDA graph、TP）
    │   └── sequence.py       # 请求序列（Sequence）管理
    ├── layers/               # 推理专用算子（TP-aware）
    │   ├── attention.py      # PagedAttention
    │   ├── linear.py         # 列/行并行 Linear
    │   ├── layernorm.py
    │   ├── activation.py
    │   ├── rotary_embedding.py
    │   ├── sampler.py
    │   ├── spec_decode.py    # 投机解码接受-拒绝（确定性草稿特例，CPU 可单测）
    │   └── embed_head.py
    ├── models/
    │   ├── __init__.py       # model_type 注册与分发（qwen3 / qwen3_moe / llama / qwen2）
    │   ├── qwen3.py          # Qwen3 稠密模型定义
    │   ├── qwen3_moe.py      # Qwen3 MoE（专家内 TP + 专家并行 EP）
    │   └── llama.py          # LLaMA / Qwen2 共用实现（同构家族，无 q/k norm）
    ├── entrypoints/
    │   └── openai/
    │       └── api_server.py # OpenAI 兼容 HTTP 服务（FastAPI + SSE）
    └── utils/
        ├── context.py        # 推理上下文（prefill/decode 布局信息）
        ├── model_download.py # hub 模型解析与自动下载（魔搭 / HF）
        └── loader.py         # 权重加载
```

## 安装

环境要求：

* Python 3.10 – 3.12
* GPU：NVIDIA（CUDA）或 AMD（ROCm，见 [ROCm / AMD GPU 部署](#rocm--amd-gpu-部署)）

```bash
git clone https://github.com/jijic137/Z-vllm.git
cd Z-vllm
pip install -e .
```

依赖由 pip 自动解析安装：`torch>=2.4`、`triton>=3`、`transformers>=4.57`、`xxhash`。
flash-attn 是可选加速后端（`pip install "z-vllm[flash]"`）：缺失时 attention 自动退回
SDPA 兜底后端（正确性不受影响，decode 吞吐略低）。
启动 OpenAI 服务需额外安装：`pip install "z-vllm[serve]"`（fastapi + uvicorn）。

### ROCm / AMD GPU 部署

代码路径对 CUDA / ROCm 是同一套：`torch.distributed` 的 `nccl` 后端在 ROCm 上自动
映射为 RCCL，无需改代码。实际差异集中在两点：

1. **flash-attn 可选**。ROCm 下可按 [flash-attention](https://github.com/Dao-AILab/flash-attention)
   官方指引自行编译，或直接不装——attention 自动退回 SDPA 兜底后端。因 SDPA decode
   需要 host 端 `context_lens.max()` 同步，与 CUDA graph 捕获不兼容，此时代码会自动
   切换 `enforce_eager=True`（prefill/decode 结果正确，decode 吞吐略低）。
2. **模型下载走魔搭**。国内网络下 `model_source="auto"`（默认）会优先从
   [ModelScope](https://www.modelscope.cn) 下载，需要
   `pip install "z-vllm[modelscope]"`；魔搭失败才回退 Hugging Face。

建议直接基于系统 ROCm torch 建 venv（避免 pip 重复拉 CUDA 版 torch）：

```bash
python3 -m venv ~/zvllm-env --system-site-packages
~/zvllm-env/bin/pip install -e ".[modelscope]"
```

多卡（TP>1）注意事项：

* 必须以**真实脚本文件**运行（`python my_infer.py`）并带
  `if __name__ == "__main__":` 保护——多卡 worker 以 spawn 方式 re-import 主模块，
  `python -c` / stdin 输入无法定位脚本，rank 会卡死且无输出；
* 任一 rank 崩溃时，其余 rank 会阻塞在 RCCL 集合通信上，需手动清理
  （`pkill -9 -f` 精确匹配你的进程）。

## 模型下载

`LLM` 的 `model` 参数除了本地目录，也支持直接传 hub 模型 ID（如 `Qwen/Qwen3-0.6B`），
首次运行自动下载并缓存到 `~/.cache`，后续运行直接命中缓存：

```python
llm = LLM("Qwen/Qwen3-0.6B")    # 自动下载；来源由 model_source 控制
```

`model_source` 取值（默认 `auto`）：

* `auto`：本地目录直接使用；模型 ID 则优先魔搭（ModelScope）、失败后回退 Hugging Face
* `modelscope` / `hf`：只走指定来源

也可以手动下载后传本地路径（以 HF 为例）：

```bash
pip install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

完整示例见 `example.py`（模型经 argv 传入：`python example.py [model]`，默认 `Qwen/Qwen3-0.6B`）。API 风格对齐 vLLM：

```python
from zvllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Z-vLLM."], sampling_params)
print(outputs[0]["text"])
```

流式输出（`stream=True` 返回生成器，逐 token 产出事件）：

```python
for event in llm.generate(["Hello, Z-vLLM."], sampling_params, stream=True):
    if event["delta"]:
        print(event["delta"], end="", flush=True)
    if event["finished"]:
        print(f"\n[finish_reason: {event['finish_reason']}]")
        break
```

### 主要参数

`LLM(model, **kwargs)`，其中 `model` 为本地模型目录或 hub 模型 ID（见上文模型下载）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_num_batched_tokens` | 16384 | 每个迭代最多调度的 token 数（chunked prefill 预算） |
| `max_num_seqs` | 512 | 最大并发请求数 |
| `max_model_len` | 4096 | 最大上下文长度（自动不超过模型 `max_position_embeddings`） |
| `gpu_memory_utilization` | 0.9 | KV cache 可占用的显存比例 |
| `tensor_parallel_size` | 1 | 张量并行卡数（1–8） |
| `model_source` | "auto" | 非本地模型 ID 的权重来源：auto（魔搭优先→HF）/ modelscope / hf |
| `weight_bits` | 16 | 权重量化：16（bf16，默认）/ 8（int8 加载时量化，per-group 128 对称，kernel 内反量化；int4 暂不支持） |
| `spec_decode` | "off" | 投机解码：off（默认）/ ngram（零训练 n-gram 草稿） |
| `spec_gamma` | 4 | 每步最多草稿 token 数（1–8，仅 ngram 生效） |
| `spec_ngram` | 4 | n-gram 匹配长度（2–8，仅 ngram 生效） |
| `enforce_eager` | False | True 时禁用 CUDA graph |
| `kvcache_block_size` | 256 | 每个 KV 块的 token 数（须为 256 的倍数） |
| `moe_tp_size` | 自动推导 | MoE 专家内 TP 卡数；缺省 = 卡数 / `moe_ep_size`（即纯 EP），显式指定可组合混合 TP×EP |
| `moe_ep_size` | 1 | MoE 专家并行卡数；须满足 `moe_tp_size * moe_ep_size == tensor_parallel_size` |
| `max_graph_bs` | 512 | CUDA Graph 捕获的最大 batch size |
| `master_port` | 2333 | 多卡 NCCL 进程组初始化端口 |
| `shm_name` / `shm_size` | `"zvllm"` / 2^20 B（约 1MB） | 多卡方法调用的共享内存名与大小 |

`SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=None, max_tokens=64, ignore_eos=False)`：

* `temperature`：0 表示贪心解码（argmax），> 0 为随机采样
* `top_k`：只保留概率最高的 k 个 token，-1 不限制
* `top_p`：nucleus 采样，保留累积概率首次达到 p 的最小前缀集
* `seed`：给定后该序列的随机流可复现
* `ignore_eos`：遇到 EOS 仍继续生成到 `max_tokens`（`bench.py` 中用于压测）

`llm.generate(prompts, sampling_params, use_tqdm=True, stream=False)`：`prompts` 支持 `str` 或 token id 列表；`sampling_params` 可为单个参数或逐请求列表。`stream=False` 按输入顺序返回完整结果（含 `"text"` / `"token_ids"` 字段）；`stream=True` 返回生成器，逐请求逐 token 产出事件（`"index"` / `"delta"` / `"text"` / `"token_ids"` / `"finished"` / `"finish_reason"`）。

### MoE 模型（Qwen3-30B-A3B）

`example_qwen3_moe.py` 展示两种典型并行方式（`moe_tp_size * moe_ep_size` 必须等于卡数）：

```python
llm = LLM("Qwen/Qwen3-30B-A3B", tensor_parallel_size=8)               # 纯专家内 TP
llm = LLM("Qwen/Qwen3-30B-A3B", tensor_parallel_size=8, moe_ep_size=8)  # 纯 EP（小 GEMM 更少，推荐）
```

MoE 模型自动禁用 CUDA graph（强制 `enforce_eager`）。

## 投机解码

零训练 n-gram 投机解码：草稿取自序列自身历史（PLD），目标模型一次 forward 对草稿做接受-拒绝验证——没有草稿模型、没有额外 draft forward。

```python
llm = LLM("Qwen/Qwen3-0.6B", spec_decode="ngram", spec_gamma=4, spec_ngram=4)
```

机制：调度在每个 decode 步查序列自身历史的尾部 n-gram 重复匹配，取至多 γ 个草稿 token（零额外 forward）；验证步把 `[last_token, 草稿…]` 回喂进一次 varlen forward，LM head 对带草稿序列保留整段 1+γ 行 logit（非投机序列仍只保留末行，批内无草稿时走"仅末行"快速路径）；接受-拒绝按 Leviathan et al. 2023 确定性草稿特例——贪心退化为 argmax 匹配，随机采样以概率 p(x) 接受草稿、首次拒绝时从残差分布补一个 bonus token，每步输出分布严格等于目标分布，投机不改变任何质量属性。

接受率统计：`engine.spec_stats`（`proposed_tokens` / `accepted_tokens` / `steps`，接受率 = accepted / proposed）。

### 正确性验证（A/B 门槛）

`ab_spec_greedy.py`（GPU 门槛，Qwen3-0.6B / bf16 / 贪心 / γ=4 / n=4，5 个进程级隔离子进程）：

| 判据 | 口径 | 结果 |
|---|---|---|
| A | 单序列 off vs on 逐位一致（硬门槛） | PASS（128 tokens 逐位一致） |
| B | 3 序列逐 prompt：未用草稿必逐位一致；用过草稿时首个分歧点两侧决定行须近 tie（min(gap_top2) ≤ 0.5）且此前无 argmax 早翻 | PASS（见下） |
| C | on3b 跨进程重跑 vs on3 逐位一致（确定性，硬门槛） | PASS（3/3 prompts） |

B 的细节：prompt 0 首个分歧 at 29，此前逐位一致；分歧点两侧决定行均为 4 路精确 bf16 tie（top2 gap = 0.0000，min_gap 0.0 ≤ 0.5）：

| 侧 | top-5 logit（分歧点决定行） |
|---|---|
| off | 2701:17.000  96934:17.000  1156:16.875  5128:16.875  1555:16.375 |
| on | 1156:17.000  2701:17.000  5128:17.000  96934:17.000  1555:16.375 |

ulp 级漂移翻转 tie-break 即产生分歧。prompts 1/2 即使用了草稿也逐位一致。

根因（B2 实验链）：off 侧自身批形状变化不引入漂移（M=3 vs M=1 逐位一致）；KV 漂移是移动前沿——首个草稿命中（产出序号 19）之前的位置逐位零漂移，之后每个新写位置带 ≤1 个 bf16 ulp 漂移、旧位置永久一致，无块错读/mask 错位；分歧后 97 个 argmax 分歧全部 ≥ 分歧点（级联）。业界对照：vLLM 等也不保证 bf16 下 spec on/off 逐位一致（验证行与 decode 行批形状不同是设计使然）。

口径：单序列 A/B 逐位一致为硬门槛（该栈 verify M=5 vs decode M=1 逐位相同，实测 3/3 绿）；多序列改用近 tie 容差门槛，避免把 bf16 固有数值行为误判为 bug。

### 性能（bench_spec.py）

`bench_spec.py`：Qwen3-0.6B，1× W7900D（SDPA 兜底 + eager），贪心，8 并发 × 192 输出 token，
随机 token-id prompt（128–300 token），已排除 warmup：

| 指标 | spec off | spec on | 加速比 |
|---|---|---|---|
| TTFT mean (ms) | 100.3 | 100.5 | 1.00 |
| TTFT p95 (ms) | 100.3 | 100.5 | 1.00 |
| TPOT mean (ms) | 30.75 | 23.31 | **1.32** |
| TPOT median (ms) | 30.75 | 18.38 | **1.67** |
| 聚合吞吐 (tok/s) | 257.1 | 193.1 | 0.75 |

接受率 0.958（proposed 889 / accepted 852，231 个 spec 步）；wall 5.97 s → 7.95 s。

如实说明：逐序列 inter-token 延迟（TPOT）确实下降，但该配置下聚合吞吐降至 0.75×。机制：
① 批内任一序列带草稿，整批即走 varlen forward（验证步每序列至多 1+γ 行），全批的
decode 快路径被更慢的路径替代；② 0.6B 的 decode 步本身很短（30.75 ms/step @ batch 8），
小模型摊不平验证开销；③ 各序列接受度漂移（TPOT mean 23.31 vs median 18.38），wall 由最慢
序列决定（≈41 ms/token，比 off 的 30.75 还慢）——低接受序列在昂贵的 varlen 步里只拿到
1 个 token。投机解码的收益区间是 decode 步长的场景（大模型 / 低并发单流），小模型高并发
恰是其最差配置。

## DP（多副本）推理

进程级多副本：把机器上的 GPU 分成 D 组，每组独立拉起一个完整的 LLMEngine 副本（独立进程组、独立 KV 块池、独立调度器）；父进程（`DPClient`）负责 round-robin 派发，并每轮给所有在途副本同步发 step（各副本 GPU forward 时间上重叠），每副本 reader 线程把 token 事件翻回调用方。副本内部仍可任意组合 TP × EP（MoE EP 通信沿用既有补零 all_reduce 方案）。

```python
from zvllm.engine.dp_engine import DPClient

client = DPClient("Qwen/Qwen3-30B-A3B", dp_size=2, tensor_parallel_size=2,
                  gpus=[2, 3, 4, 5], moe_ep_size=2, master_port=2345)
client.generate(["..."])    # 与 LLMEngine 同构接口：add_request / generate / generate_stream / abort_request
```

服务层：`api_server.py --dp-size 2 --dp-gpus 2,3,4,5`，OpenAI 兼容接口不变（单引擎路径零改动）。

设计要点：

* **进程级隔离**：每副本先设 `CUDA_VISIBLE_DEVICES` 再建 LLM，进程组、NCCL 端口（master_port+r）、共享内存名（`{shm}_dp{r}`）天然错开；副本间不共享内存，单副本崩溃不污染其他副本状态。
* **父进程路由 + 驱动**：driver 线程 round-robin 派发新请求，每轮同时给所有在途副本发 step；per-replica reader 线程做 seq_id→req_id 翻译，事件经 `on_event` 回推（锁外调用）。
* **不变式**：每副本至多一个未回话 step（任何回话清零 outstanding）；seq→req 映射仅终态摘除（中间态误摘会丢事件，是实测修掉的第一 bug）；每请求末事件恒为 finished（skip/abort/fatal 均补发）。
* **容错**：per-replica 看门狗（step_timeout 无回话 → 全局 fail-fast：拒新请求 + 在途请求快速失败 + `on_fatal` 回调）；api_server 将 fatal 转为 HTTP 503 并拒绝新请求。
* **与 vLLM 的形态差异**：vLLM 的 DP attention 是"同进程内 DP"（一个模型实例，attention DP rank 间共享调度、专家 EP rank 间 all-to-all 换 token）；本实现是"进程级多副本"（每副本完整独立引擎，请求粒度 round-robin 路由，副本间零通信）。形态更简单、隔离更强（单副本崩溃不拖垮全局，看门狗 + fail-fast 可控），代价是每副本独立持有权重与 KV：无跨副本 prefix 共享、无 token 粒度负载均衡。

### 性能（bench_dp.py）

硬件与负载：4× W7900D 48GB（gfx1100，ROCm 7.1.1），Qwen3-30B-A3B（bf16），
prompt 256 随机 token / 生成 128 token，N≥16，并发 C ∈ {1, 4, 16, 32, 64} 个 worker 线程，
warmup 1 请求已排除，`enforce-eager`（SDPA 兜底、无 CUDA graph）；A 组 2026-08-20 13:25–13:40 测得，
B 组 13:42 起补跑（约 14:00 完成），两组时间错开、非严格交错；A 组采于 DPClient daemon 修复之前，
但 A 路径（独立 LLMEngine）不涉及该修复，数字不受影响。

A = 单引擎 tp4×ep4（4 卡 1 个引擎，EP 更宽）；B = DP 2×tp2×ep2（2 副本，各自独立 KV 块池与调度器）：

| 并发 C | 形态 | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | 吞吐 (tok/s) |
|---|---|---|---|---|---|
| 1 | A | 134.3 | 137.3 | 105.33 | 9.5 |
| 1 | B | 204.5 | 393.6 | 101.99 | 9.7 |
| 4 | A | 746.7 | 1181.0 | 105.34 | 36.2 |
| 4 | B | 651.8 | 1353.3 | 108.26 | 34.6 |
| 16 | A | 1522.5 | 1523.6 | 112.94 | 129.1 |
| 16 | B | 2341.1 | 3258.3 | 122.11 | 111.2 |
| 32 | A | 2341.0 | 2342.1 | 123.27 | 227.6 |
| 32 | B | 2757.0 | 3680.5 | 132.04 | 204.8 |
| 64 | A | 3647.2 | 4895.8 | 160.86 | 339.3 |
| 64 | B | 3815.4 | 4599.3 | 143.29 | 365.2 |

> vLLM 对比口径：vLLM 的 ROCm 官方支持仅覆盖 CDNA 系列，本机 gfx1100（RDNA3）无可行 vLLM 路径
>（本机 vllm venv 实测 import 失败，阻塞证据已留档），故本章为同一代码库内两种并行形态的 A/B 对比
> + 与 vLLM DP attention 的结构性差异说明，不呈现 vLLM 数字。

**机制分析**

* **低并发（C=1）**：B 的 TPOT 低 3.2%（102.0 vs 105.3 ms），但 TTFT mean 204.5 vs 134.3 ms 且分布更宽
  （median 142.0 / p95 393.6，A 的 p95 为 137.3）：顺序请求下两副本交替 prefill，round-robin 派发与
  per-replica 进程/通信开销未被摊薄。
* **中并发（C=16/32）**：A 明显占优（吞吐 +16.1% / +11.1%）。两形态每 rank 的 MoE 计算量结构上相同
  （T×topk/EP 恒定），差异在稠密部分：A 的集中批（16/32 序列）GEMM 算术强度更高，且 tp4 每 rank 读的
  KV head 更少（2 vs 4）。
* **高并发（C=64）**：B 反超（吞吐 +7.6%，TPOT −10.9%）。A 单引擎批到 64，步延迟 C=32→64 从
  123.3→160.9 ms（+30.5%，B 仅 +8.5%），TTFT p95 4895.8 vs 4599.3——eager + SDPA 兜底（无 CUDA graph）
  下逐层开销 ×48 放大了批与 4-rank all_reduce（流量为 B 的 2 倍）的代价；B 双调度器把批拆半、2-rank
  通信、两副本 forward 时间重叠，步延迟增长更缓，交叉点落在 C=32–64。
* **结论**：两形态总 KV 容量相同（A：4 卡分片池；B：2×2 卡独立池），高并发差异来自批拆分与通信形态
  而非容量：C≤32 选单引擎宽 EP，逼近单引擎并发上限（C≥64）时 DP 拆分发力。两者互补，亦可叠加
  （D×tp×ep 任意组合）。

## OpenAI 兼容服务

```bash
pip install "z-vllm[serve]"
python -m zvllm.entrypoints.openai.api_server \
    --model ~/huggingface/Qwen3-0.6B/ --port 8000
```

提供 `GET /health`、`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/completions`（均支持 `stream`，SSE 推送）。可直接对接 OpenAI SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

命令行参数与引擎参数一一对应（见[主要参数](#主要参数)），如 `--weight-bits 8` 启动 int8 量化模型。

`stop` 已支持：生成文本以任一停止串结尾即终止（`finish_reason="stop"`，停止串保留在输出中）。`presence_penalty` / `frequency_penalty` 被接受但暂不生效。流式请求客户端断连时自动取消对应序列并释放其 KV 块（服务日志出现 `aborted seq N`）。单步 GPU 阶段超过 60 s 看门狗超时时，在途请求快速失败、进程退出，交由上层重启。

## Benchmark

基准脚本见 `bench.py`。基线数据：

**测试配置**

* 硬件：RTX 4070 Laptop (8GB)
* 模型：Qwen3-0.6B
* 请求数：256 条
* 输入/输出长度：100–1024 token 随机采样

| 推理引擎 | 输出 token 数 | 耗时 (s) | 吞吐 (tokens/s) |
|---|---|---|---|
| vLLM | 133,966 | 98.37 | 1361.84 |
| Z-vLLM | 133,966 | 93.41 | 1434.13 |

## 已验证

以下环境与配置已实测通过：

* **单卡（CUDA）**：RTX 4070 Laptop + Qwen3-0.6B，见 [Benchmark](#benchmark)
* **ROCm（AMD）**：2 × W7900D 48GB，ROCm 7.1 / torch 2.11（rocm 构建）。代码路径零改动
  （`nccl` 后端自动映射 RCCL），未装 flash-attn 时自动 SDPA 兜底 + `enforce_eager=True`，
  权重经魔搭社区下载
* **AMD 单卡稠密吞吐**：Qwen3-0.6B，1× W7900D，SDPA 兜底 + eager，256 请求 / 133,966 输出 token，
  聚合吞吐 480.21 tok/s（278.97 s）；同一 256 请求集下上游 CUDA flash-attn + CUDA graph 配置为
  1434 tok/s（见 [Benchmark](#benchmark)），差距来自 attention 后端与图模式
* **更多模型家族（Qwen2 / LLaMA）**：2026-08-20，1× W7900D，SDPA 兜底 + eager，单流生成
  * Qwen2-0.5B（权重经魔搭，`tie_word_embeddings=true`，覆盖 tied lm_head 路径）：decode ≈35–36 tok/s，输出连贯
  * Llama-3.2-1B-Instruct（魔搭镜像 `LLM-Research/Llama-3.2-1B-Instruct`）：decode ≈59–62 tok/s（prefill 41–82），
    生成质量检查通过（自报身份正确、100 以内 25 个素数列表完整无误、正常 EOS 结束）
  * Qwen3-0.6B 回归：≈67 tok/s（此前验证区间 35–69），模型分发 / RMSNorm 改动无回归
  * CPU 单测：llama / qwen2 配置解析（eps / rope_theta 流通过）、权重加载布局、全 forward 对拍朴素参考
    （max_diff 4.29e-6 / 6.56e-6）、tied embeddings、model_type 分发
* **OpenAI 服务真机 e2e**：2026-08-20，1× W7900D + Qwen2-0.5B（魔搭权重）——stop 流正常收尾
  （`finish_reason="stop"` + `[DONE]`）；客户端 3 s 后断连 → 服务日志出现 `aborted seq`（序列取消、KV 释放）；
  服务保持健康，后续请求正常
* **MoE 专家并行（EP）**：Qwen3-30B-A3B（128 专家 / 48 层全 MoE / bf16）
  * TP=2 / 4 / 8 均端到端跑通；纯 EP（`moe_ep_size=N`）与专家内 TP（`moe_ep_size=1`）均正确生成
  * CPU 单测：EP 路径 MoE 前向与逐 token 参照一致（bf16，max_diff 0.0）；
    Triton 融合路径与 bmm 兜底路径位级一致（max_diff 0.0）
  * TP 超过 KV head 数（8 卡跑 4 KV head 模型）时复制 KV head，
    GQA 连续映射经权重级单测校验（每 rank 取到与其 q head 对应的 kv head）
  * 专家权重以原生 stacked 大 buffer 存储（各专家权重是 buffer 的 view），
    融合路径零拷贝、零额外显存
  * 同一配置跨进程重复运行输出逐字节可复现；EP=2/4/8 输出文本一致
    （差异仅来自 bf16 求和顺序），均为连贯正确的英文输出
  * 并发 decode 扫描（`bench_moe_conc.py`）：1/16/32/64 请求 × EP=2/4/8，
    聚合吞吐随 N 近线性（每翻倍 ≈1.93–1.99×），峰值 587.79 tok/s（EP=8，N=64）；
    T=1 平台期在高并发下被击穿，EP=8 反超（N=64 时 +10.8%）
* **权重量化（W8A16）**：2026-08-20，`weight_bits=8`（加载时 int8，per-group 128 对称，kernel 内反量化；embed/lm_head 保持 bf16）
  * 每卡纯权重显存（loader 级实测，不含 KV cache）：Qwen3-30B-A3B EP=2，bf16 28.5 GiB → W8 14.8 GiB（−48%）
  * **单张 48GB 卡跑 30B MoE**：W8 TP=1 权重 29.5 GiB + KV cache，单流贪心 30.7 tok/s、输出连贯（bf16 57GB 单卡放不下，至少 2 卡）
  * 精度：Qwen3-0.6B ΔPPL +0.0845（≤ 0.1）；30B 贪心 A/B（W8 vs bf16）主体一致、仅量化噪声级分歧
  * kernel 对拍：稠密 `quant_linear` 相对误差 ≤ 2.1e-3（vs 物化反量化 mm）；MoE Triton QUANT_W 分支与 bf16 分支位级一致（max_diff 0.0）
  * 速度（30B-A3B，4 prompt × 64 token 混合）：EP=2 W8 26.0 vs bf16 32.7 tok/s（小批量反量化开销，如实记录）；单卡 W8 30.7 tok/s
* **投机解码（n-gram）**：2026-08-20，1× W7900D + Qwen3-0.6B（bf16，γ=4 / n=4，贪心）
  * CPU 单测：调度器级（草稿获取 / 验证步记账 / 块预留 / 接受回写 / 统计）+ 全链路仿真（真实模型 + 真实 ModelRunner，确定性目标函数下 spec on/off 必须逐位一致；覆盖 chunked prefill / 混合批 / 抢占 / prefix cache）；全套 93 项全绿
  * 贪心 A/B 门槛（`ab_spec_greedy.py`）：单序列 off/on 逐位一致（128 tokens）；3 序列：prompt 0 首个分歧 at 29，两侧决定行均为 4 路精确 bf16 tie（gap 0.0，min_gap ≤ 0.5 NEAR-TIE），分歧前无 argmax 早翻；prompts 1/2 逐位一致（含用草稿的情况）；on3b 跨进程重跑逐位一致 → ABG_PASS
  * 多序列分歧根因（B2 实验）：bf16 近 tie 固有数值行为而非 bug——off 批形状本身无漂移（M=3 vs M=1 逐位一致）、KV 漂移自首个草稿命中起（移动前沿，旧位置永久一致）、根因行 4 路精确 tie（gap 0.0）、97 个 argmax 分歧全部 ≥ 分歧点（级联）
  * 性能（`bench_spec.py`，8 并发 × 192 tok 贪心，随机 token prompt）：TPOT mean 30.75 → 23.31 ms（1.32×）、median 18.38 ms（1.67×）；TTFT 不变（100.3 ms）；接受率 0.958（852/889，231 spec 步）；聚合吞吐 257.1 → 193.1 tok/s（0.75×，wall 5.97 → 7.95 s）——批内任一带草稿即全批 varlen、小模型 decode 步太短摊不平验证开销、最慢序列（≈41 ms/token）决定 wall，如实记录；收益区间在大模型 / 低并发单流（decode 步长）

* **DP 多副本（DP 2×tp2×ep2）**：2026-08-20，4× W7900D（GPU 2–5）+ Qwen3-30B-A3B bf16，`bench_dp.py` 5 并发点扫描
  * C=64：吞吐 365.2 tok/s vs 单引擎 tp4×ep4 339.3（+7.6%），TPOT 143.3 vs 160.9 ms（−10.9%）；C≤32 单引擎占优
    （C=16 +16.1% / C=32 +11.1%），交叉点 C=32–64，机制分析见 [DP（多副本）推理](#dp多副本推理)
  * 实测发现并修复：daemon 副本无法 spawn TP rank 子进程（AssertionError）→ 改非 daemon + 父进程存活检查，
    父死副本自退（防 SIGKILL 孤儿）；shm 名与同机既有服务冲突 → `--shm-name`
  * tpot 自洽检查全过（per-req 重算 vs json 差 <0.5 ms）；CPU 101 测试全绿

性能数据（Qwen3-30B-A3B，W7900D，SDPA 兜底，单请求，prompt ≈ 10 token / 生成 64 token，贪心；
MoE 数字来自 `bench_moe_ep.py`，best of 2 runs）：

| 并行模式 | 耗时 (s) | 吞吐 (tokens/s) |
|---|---|---|
| 纯 EP TP=2（优化前基线，逐专家 gather 循环） | 17.3 – 17.4 | 3.7 |
| 纯 EP TP=2（Triton grouped-GEMM 融合） | 6.57 | 9.74 |
| 纯 EP TP=4（Triton 融合） | 6.75 | 9.49 |
| 纯 EP TP=8（Triton 融合，KV head 复制） | 6.68 | 9.58 |
| 专家内 TP TP=2（`moe_ep_size=1`） | 29.5 – 30.6 | 2.1 |

> 注：MoE decode 阶段经自研 Triton grouped-GEMM 融合（排序分桶 → 两个 grouped GEMM →
> silu&mul → scatter → all_reduce）后，单层 MoE 子层 3.65 ms → 0.76 ms（≈4.8×），
> 端到端 EP=2 3.7 → 9.74 tok/s（约 2.6×）。EP≥2 后吞吐进入平台期：T=1 单请求下
> 每步延迟由 attention 与逐层 all_reduce 主导，MoE 子层已非主要瓶颈——
> 该平台期只在 T=1 小批量区间成立，N≥16 并发后 EP 收益重新出现（见下表交叉现象）。
> 与 [Benchmark](#benchmark) 的稠密模型多请求批处理数字不可直接对比。

并发 decode 数据（同模型同硬件；N 路贪心并发，每流 prompt ≈ 10 token / 生成 64 token，
来自 `bench_moe_conc.py`，best of 2 runs；decode 步延迟为稳态段中位数）：

| 并行模式 | 并发 N | decode 步延迟 (ms) | 聚合吞吐 (tokens/s) | 单流吞吐 (tokens/s) |
|---|---|---|---|---|
| 纯 EP TP=2 | 1 | 101.4 | 10.02 | 10.02 |
| 纯 EP TP=2 | 16 | 121.4 | 136.83 | 8.55 |
| 纯 EP TP=2 | 32 | 122.7 | 269.95 | 8.44 |
| 纯 EP TP=2 | 64 | 124.1 | 530.56 | 8.29 |
| 纯 EP TP=4 | 1 | 104.7 | 9.68 | 9.68 |
| 纯 EP TP=4 | 16 | 111.0 | 146.84 | 9.18 |
| 纯 EP TP=4 | 32 | 112.7 | 290.42 | 9.08 |
| 纯 EP TP=4 | 64 | 117.0 | 560.75 | 8.76 |
| 纯 EP TP=8 | 1 | 111.1 | 9.13 | 9.13 |
| 纯 EP TP=8 | 16 | 109.4 | 148.19 | 9.26 |
| 纯 EP TP=8 | 32 | 109.6 | 296.27 | 9.26 |
| 纯 EP TP=8 | 64 | 110.5 | **587.79** | 9.18 |

> 注：聚合吞吐随 N 近线性增长（每翻倍 ≈1.93–1.99×）。**EP 交叉现象**：N=1 时
> EP=2 最快（10.02 > 9.13，平台期），N≥16 后 EP=8 反超（N=64 时 587.8 vs EP=2 530.6，
> +10.8%）——每 rank 的 MoE 计算量按 T×topk/EP 随并发增长，T 足够大后成为主导项，
> 而每步 all_reduce 流量是完整 [T, topk, H] 补零张量，随 T 增长但不随 EP 变化。
> 单流吞吐并发 1→64 时 EP=2/4 降 17%/10%，EP=8 持平；步延迟 EP=2 +22% / EP=4 +12% / EP=8 持平。
> 口径说明：日志含 `rms_forward` 的 torch.compile 重编译告警（达重编译上限后回退 eager），
> best-of-2 + 稳态中位指标保证数字不受影响。

## Roadmap

- [x] 支持更多模型家族（LLaMA、Qwen2 等）（llama.py 同构共用实现 + model_type 注册分发，Qwen2-0.5B / Llama-3.2-1B 真机验证）
- [x] 权重量化 W8A16（加载时 int8 per-group 128 + kernel 内反量化 GEMM；30B-A3B 每卡权重显存 28.5 → 14.8 GiB，单张 48GB 卡跑 30B MoE，见[已验证](#已验证)）
- [ ] W4 量化（加载 AWQ / GPTQ 离线量化 checkpoint，暂缓）
- [x] 逐请求取消与停止串（stop strings）（服务层 stop 直通 + 断连自动取消 + GPU 阶段看门狗）
- [x] MoE decode 性能优化（Triton grouped-GEMM 融合路径，EP=2 3.7 → 9.74 tok/s，约 2.6×）
- [x] 投机解码（零训练 n-gram 草稿 + 验证步接受-拒绝，输出分布不变；贪心 A/B 一致性门槛 + 性能基准，见[投机解码](#投机解码)）
- [x] DP 多副本推理（进程级 D×TP 副本 + round-robin 路由 + 同步 step + 看门狗 fail-fast；4 卡 30B-A3B A/B：C≤32 单引擎占优，C=64 DP 反超 +7.6%，见[DP（多副本）推理](#dp多副本推理)）
- [ ] ROCm 加速：flash-attn ROCm 编译、CUDA graph 兼容性

## 来源与许可

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（Xingkai Yu，MIT License）二次开发，原始版权声明保留在 `LICENSE` 文件中。
