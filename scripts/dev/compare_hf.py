"""开发用：小模型上对拍 zvllm 与 HF transformers 的贪心输出（验证实现正确性）。

权重随机的小模型（如 PrimeIntellect/qwen3-moe-tiny）也能当验证靶：随机权重不影响
力学判据——只要两端在同一 prompt 上给出同样的贪心 token 序列，就说明
RMSNorm / RoPE / GQA / 路由 / MoE 组合与 HF 参考实现一致。

用法：python scripts/dev/compare_hf.py PrimeIntellect/qwen3-moe-tiny --max-tokens 16

为什么拆成子进程：zvllm 按 gpu_memory_utilization 预分配 KV cache（默认占 90% 显存），
同进程内 `del llm` 断不掉引擎里的引用环，紧接着再加载 HF 模型会超出 11GB 触发换页，
表现为 GPU 100% 但长时间无输出。因此两个实现各自在独立子进程里跑，父进程只做比对。

贪心逐 token 会放大浮点差异，末端近 tie 处可能分叉。脚本在分叉时用 HF 前向复核
该位置的 logits 排名与 margin：若 zvllm 选中的 token 在 HF 中排名靠前且 margin 极小，
即为"近 tie 分叉"（与本仓库 ab_spec_greedy.py 的判据一致），不算实现错误。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MARKER = "##RESULT##"

# 2080 Ti（sm_75，11GB）上 inductor 会开多个 compile worker 吃满显存且长时间不返回，
# 对拍必须走纯 eager；torch.compile 相关的验证另开脚本做。
CHILD_ENV = {
    **os.environ,
    "TORCHDYNAMO_DISABLE": "1",
    "TQDM_DISABLE": "1",
    "MODELSCOPE_LOG_LEVEL": "40",
    "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
}


def run_child(args: list[str], timeout: int = 900) -> dict:
    """在独立子进程里跑一侧，解析它打出的 MARKER json 行。"""
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), *args],
                          capture_output=True, text=True, env=CHILD_ENV, cwd=ROOT, timeout=timeout)
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            payload = json.loads(line[len(MARKER):].strip())
    if payload is None:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise RuntimeError(f"子进程未产出结果（returncode={proc.returncode}）：\n{tail}")
    return payload


def side_zvllm(args) -> dict:
    import torch

    from zvllm import LLM, SamplingParams

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem)
    outputs = llm.generate([args.prompt], SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                           use_tqdm=False)
    ids = outputs[0]["token_ids"]
    print(f"{MARKER} {json.dumps({'ids': ids})}", flush=True)
    del llm, outputs, torch


def side_hf(args) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from zvllm.utils.model_download import resolve_model_path

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    # dtype="auto" 跟随 config（zvllm 侧也是 set_default_dtype(hf_config.dtype)），两边同 dtype 才可比
    model = AutoModelForCausalLM.from_pretrained(path, dtype="auto").to("cuda").eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens,
                                   use_cache=True, pad_token_id=tokenizer.eos_token_id)
    ids = generated[0][inputs["input_ids"].shape[1]:].tolist()

    diag = ""
    ours = json.loads(args.ours_ids) if args.ours_ids else []
    if ours and ours != ids:
        common = 0
        for a, b in zip(ours, ids):
            if a != b:
                break
            common += 1
        if common < len(ours):
            # 用 zvllm 的前缀做 teacher forcing，复核它在该位置选了什么
            full = torch.cat([inputs["input_ids"],
                              torch.tensor([ours[:common]], device="cuda")], dim=1)
            with torch.inference_mode():
                logits = model(full).logits[0, -1].float()
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(5)
            chosen = ours[common]
            ranking = int((logits > logits[chosen]).sum().item())
            names = [tokenizer.decode([i]) for i in top_ids.tolist()]
            detail = ", ".join(f"{n!r}:{p:.4f}" for n, p in zip(names, top_probs.tolist()))
            top1, top2 = int(top_ids[0]), int(top_ids[1])
            tied = "完全相同（argmax 平局，打破方式不同）" if logits[top1] == logits[top2] \
                else f"相差 {abs(logits[top1] - logits[top2]):.6f}"
            diag = (f"HF top-5 @分叉位: {detail}\n"
                    f"HF 前两名原始 logit: {tokenizer.decode([top1])!r}={logits[top1]:.6f}, "
                    f"{tokenizer.decode([top2])!r}={logits[top2]:.6f} -> {tied}\n"
                    f"zvllm 选中 {chosen!r}（HF 中严格大于它的 logit 有 {ranking} 个）")
    print(f"{MARKER} {json.dumps({'ids': ids, 'diag': diag})}", flush=True)
    del model, tokenizer, torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    # 内部参数：子进程模式下分别跑单侧
    parser.add_argument("--side", choices=["zvllm", "hf"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ours-ids", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.side == "zvllm":
        side_zvllm(args)
        return
    if args.side == "hf":
        side_hf(args)
        return

    common_args = [args.model, "--prompt", args.prompt, "--max-tokens", str(args.max_tokens)]
    ours = run_child(["--side", "zvllm", *common_args, "--gpu-mem", str(args.gpu_mem)])["ids"]
    theirs_payload = run_child(["--side", "hf", *common_args,
                                "--ours-ids", json.dumps(ours)])
    theirs, diag = theirs_payload["ids"], theirs_payload["diag"]

    print(f"\nprompt: {args.prompt!r}")
    print(f"zvllm ids: {ours}")
    print(f"hf    ids: {theirs}")
    if ours == theirs:
        print("\nRESULT: 全序列一致（贪心 token 完全相同）")
        return
    common = 0
    for a, b in zip(ours, theirs):
        if a != b:
            break
        common += 1
    print(f"\nRESULT: 前 {common} 个 token 一致后在位置 {common} 分叉"
          f"（zvllm={ours[common] if common < len(ours) else None}, "
          f"hf={theirs[common] if common < len(theirs) else None}）")
    if diag:
        print(diag)


if __name__ == "__main__":
    main()
