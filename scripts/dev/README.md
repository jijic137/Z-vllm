# 开发验证工具

这些脚本是"功能验证口径"的载体：每次打 tag 时说的"验过了"，指的就是在这里跑出来的结果。
它们不参与打包（`pyproject.toml` 只收 `zvllm*`），也不依赖仓库外的任何私货。

## 环境前提

验证在 WSL2 Ubuntu 上做（Windows 侧缺 triton，且 2080 Ti 是 sm_75，flash-attn 不支持）：

```bash
python3 -m venv --system-site-packages ~/venvs/zvllm     # 复用系统已有的 torch/triton/transformers
~/venvs/zvllm/bin/pip install -e . --no-deps
~/venvs/zvllm/bin/pip install xxhash modelscope 'fastapi>=0.110' 'uvicorn>=0.29' pytest
```

实测环境：Python 3.12 / torch 2.12.0+cu126 / triton 3.7.0 / transformers 5.10.2 / RTX 2080 Ti 11GB (sm_75)。
模型走魔搭下载（`modelscope.cn` 国内直连，不必挂代理）。

## 脚本

**`download_models.py`** — 从魔搭拉验证用小模型并打印体积。

```bash
python scripts/dev/download_models.py Qwen/Qwen3-0.6B Qwen/Qwen2.5-0.5B
```

**`smoke.py`** — 加载任意受支持模型跑一次生成，用来逐条验证引擎开关。

```bash
python scripts/dev/smoke.py PrimeIntellect/qwen3-moe-tiny
python scripts/dev/smoke.py Qwen/Qwen3-0.6B --no-eager --kv-bits 8 --spec ngram
```

支持 `--tp/--ep/--weight-bits/--kv-bits/--spec/--no-eager/--gpu-mem/--prompt`。
与 `example.py` 的区别：不加 chat template（兼容 base 模型）、关进度条、开关直接透传。

**`compare_hf.py`** — 与 HF transformers 参考实现对拍贪心输出，是判断"实现有没有算错"的主判据。

```bash
python scripts/dev/compare_hf.py PrimeIntellect/qwen3-moe-tiny --max-tokens 16
python scripts/dev/compare_hf.py Qwen/Qwen3-0.6B --max-tokens 24 --gpu-mem 0.6
```

两侧各自跑在独立子进程里（zvllm 会按 `gpu_memory_utilization` 预占约 90% 显存，同进程里
`del llm` 断不掉引擎的引用环，紧接加载 HF 模型会超 11GB 触发换页，表现为 GPU 100% 但长时间无输出）。
分叉时会用 HF 前向复核该位置：打印前两名原始 logit、以及 zvllm 选中 token 的排名。

**`inspect_ckpt.py`** — 按层前缀打印权重键与形状，核对模型结构与命名（新增模型家族时先跑这个）。

```bash
python scripts/dev/inspect_ckpt.py PrimeIntellect/qwen3-moe-tiny model.layers.0.
```

## 两个环境坑

`torch.compile` / inductor 在 2080 Ti（sm_75，11GB）上会开多个 compile worker 吃满显存且长时间不返回。
跑对拍前已用 `TORCHDYNAMO_DISABLE=1` 关掉；要做 compile / CUDA graph 相关验证，建议换更大显存的卡。

`tests/test_spec_cpu_sim.py` 在本机稳定失败 4 例：这些用例号称纯 CPU，但本机 triton 可导入且 CUDA 可用，
于是拿 CPU 张量去喂 Triton kernel，报 `Pointer argument cannot be accessed from Triton (cpu tensor?)`。
属环境错配，不是实现问题（在无 triton 或无敌 CUDA 的机器上会走 CPU 路径）。

## 可用的验证靶

| 模型 | 体积 | 用途 |
| --- | --- | --- |
| `PrimeIntellect/qwen3-moe-tiny` | 1.26GB | 与 Qwen3-30B-A3B 同构的迷你 MoE（16 专家 top-4、24 层、`mlp_only_layers=[0]`），验证 MoE 路由 / 专家加载 / fused grouped-GEMM；权重随机，输出无意义，只看与 HF 是否一致 |
| `Qwen/Qwen3-0.6B` | 1.40GB | 最小的 Qwen3 稠密模型，验证主流程（PagedAttention / continuous batching / KV cache / 采样 / 流式） |
| `Qwen/Qwen2.5-0.5B` | 0.93GB | qwen2 家族（LLaMA 同构实现）分支 |

官方 Qwen MoE 最小是 Qwen1.5-MoE-A2.7B（14.3B 总参 / 2.7B 激活，bf16 26.7GB），
压到 W8 也有约 13.3GB，超出 11GB 单卡，且属 `qwen2_moe` 结构（共享专家 + 另一套 token 归一化），
本项目未实现——所以 MoE 相关开发都在迷你模型上验证。
