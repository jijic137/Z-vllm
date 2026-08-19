<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Z-vLLM

从零实现的轻量级 LLM 推理引擎，个人二次开发项目。

约 1,900 行 Python 代码，完整实现现代 LLM 推理的核心机制：PagedAttention、continuous batching、prefix caching、chunked prefill、张量并行、专家并行（MoE）、CUDA graph、torch.compile。推理吞吐与 vLLM 相当（见 [Benchmark](#benchmark)）。

## 特性

* 🚀 **快速离线推理** — 吞吐与 vLLM 相当
* 📖 **可读的代码库** — 约 1,900 行 Python，完整推理管线清晰可见
* 🖥️ **CUDA / ROCm 双平台** — 同一套代码支持 NVIDIA 与 AMD GPU，flash-attn 缺失时自动 SDPA 兜底
* 📦 **PagedAttention** — KV cache 按 256-token 块管理，显存无碎片
* 🔁 **Continuous batching** — 以迭代为粒度动态拼批
* ✂️ **Chunked prefill** — 长 prompt 分块 prefill，与 decode 交错执行
* 🧩 **Prefix caching** — 相同前缀跨请求共享，节省 prefill 计算
* 🧮 **张量并行** — 支持 1–8 卡
* 🧱 **专家并行（EP）** — MoE 专家切分到多卡，专家内 TP × EP 任意组合
* ⚡ **CUDA graph + torch.compile** — 捕获 decode 图，降低 launch 开销
* 🤖 **支持 Qwen3 稠密与 MoE 模型** — 如 Qwen3-0.6B、Qwen3-30B-A3B
* 🎲 **完整采样** — 贪心（temperature=0）、top-k、top-p、seed 可复现
* 📡 **流式输出** — `generate(..., stream=True)` 逐 token 产出事件
* 🔌 **OpenAI 兼容服务** — FastAPI + SSE，`/v1/chat/completions`、`/v1/completions` 直接对接 OpenAI SDK

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
    │   └── embed_head.py
    ├── models/
    │   ├── qwen3.py          # Qwen3 稠密模型定义
    │   └── qwen3_moe.py      # Qwen3 MoE（专家内 TP + 专家并行 EP）
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

完整示例见 `example.py`。API 风格对齐 vLLM：

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

`stop` / `presence_penalty` / `frequency_penalty` 等 OpenAI 字段被接受但暂不生效。

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
* **MoE 专家并行（EP）**：Qwen3-30B-A3B（128 专家 / 48 层全 MoE / bf16），TP=2
  * `moe_ep_size=2`（纯 EP，每卡 64 专家）与 `moe_ep_size=1`（专家内 TP=2）均正确生成
  * CPU 单测：EP 路径 MoE 前向与逐 token 参照一致（bf16，max_diff 0.0）
  * 两种模式各自跨进程重复运行输出逐字节可复现；模式间输出差异来自浮点求和顺序
    （EP 无 all-reduce vs 专家内 TP all-reduce），均为连贯正确的英文输出

性能数据（Qwen3-30B-A3B，TP=2，W7900D × 2，SDPA 兜底，单请求，prompt ≈ 10 token / 生成 64 token，贪心）：

| 并行模式 | 耗时 (s) | 吞吐 (tokens/s) |
|---|---|---|
| 纯 EP（`moe_ep_size=2`，专家内 TP 自动推导 = 1） | 17.3 – 17.4 | 3.7 |
| 专家内 TP（`moe_ep_size=1`） | 29.5 – 30.6 | 2.1 |

> 注：MoE decode 吞吐当前受逐专家 gather 循环与 host 同步（每层 64 专家 × 48 层/步）
> 限制，属预期开销而非缺陷；与 [Benchmark](#benchmark) 的稠密模型多请求批处理数字
> 不可直接对比。

## Roadmap

- [ ] 支持更多模型家族（LLaMA、Qwen2 等）
- [ ] 权重量化支持（AWQ / GPTQ）
- [ ] 逐请求取消与停止串（stop strings）
- [ ] MoE decode 性能优化（逐专家 gather / host 同步，Triton 融合）
- [ ] ROCm 加速：flash-attn ROCm 编译、CUDA graph 兼容性

## 来源与许可

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（Xingkai Yu，MIT License）二次开发，原始版权声明保留在 `LICENSE` 文件中。
