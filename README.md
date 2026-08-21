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
- [ ] 权重量化支持（AWQ / GPTQ）
- [x] 逐请求取消与停止串（stop strings）（服务层 stop 直通 + 断连自动取消 + GPU 阶段看门狗）
- [x] MoE decode 性能优化（Triton grouped-GEMM 融合路径，EP=2 3.7 → 9.74 tok/s，约 2.6×）
- [ ] ROCm 加速：flash-attn ROCm 编译、CUDA graph 兼容性

## 来源与许可

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（Xingkai Yu，MIT License）二次开发，原始版权声明保留在 `LICENSE` 文件中。
