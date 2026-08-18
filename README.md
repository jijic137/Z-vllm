<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Z-vLLM

从零实现的轻量级 LLM 推理引擎，个人二次开发项目。

约 1,900 行 Python 代码，完整实现现代 LLM 推理的核心机制：PagedAttention、continuous batching、prefix caching、chunked prefill、张量并行、专家并行（MoE）、CUDA graph、torch.compile。推理吞吐与 vLLM 相当（见 [Benchmark](#benchmark)）。

## 特性

* 🚀 **快速离线推理** — 吞吐与 vLLM 相当
* 📖 **可读的代码库** — 约 1,900 行 Python，完整推理管线清晰可见
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
        ├── context.py        # TP 通信上下文
        └── loader.py         # 权重加载
```

## 安装

环境要求：

* Python 3.10 – 3.12
* NVIDIA GPU + CUDA
* [flash-attn](https://github.com/Dao-AILab/flash-attention)（可能需按官方指引编译安装）

```bash
git clone https://github.com/jijic137/Z-vllm.git
cd Z-vllm
pip install -e .
```

依赖由 pip 自动解析安装：`torch>=2.4`、`triton>=3`、`transformers>=4.51`、`flash-attn`、`xxhash`。
启动 OpenAI 服务需额外安装：`pip install "z-vllm[serve]"`（fastapi + uvicorn）。

## 模型下载

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

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
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

`LLM(model, **kwargs)`，其中 `model` 为本地模型目录：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_num_batched_tokens` | 16384 | 每个迭代最多调度的 token 数（chunked prefill 预算） |
| `max_num_seqs` | 512 | 最大并发请求数 |
| `max_model_len` | 4096 | 最大上下文长度（自动不超过模型 `max_position_embeddings`） |
| `gpu_memory_utilization` | 0.9 | KV cache 可占用的显存比例 |
| `tensor_parallel_size` | 1 | 张量并行卡数（1–8） |
| `enforce_eager` | False | True 时禁用 CUDA graph |
| `kvcache_block_size` | 256 | 每个 KV 块的 token 数（须为 256 的倍数） |
| `moe_tp_size` | = `tensor_parallel_size` | MoE 专家内 TP 卡数（缺省即纯 TP 模式） |
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
llm = LLM(path, tensor_parallel_size=8)                  # 纯专家内 TP
llm = LLM(path, tensor_parallel_size=8, moe_ep_size=8)   # 纯 EP（小 GEMM 更少，推荐）
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

## Roadmap

- [ ] 支持更多模型家族（LLaMA、Qwen2 等）
- [ ] 权重量化支持（AWQ / GPTQ）
- [ ] 逐请求取消与停止串（stop strings）

## 来源与许可

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（Xingkai Yu，MIT License）二次开发，原始版权声明保留在 `LICENSE` 文件中。
