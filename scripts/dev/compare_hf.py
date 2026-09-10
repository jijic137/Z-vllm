"""开发用：小模型上对拍 zvllm 与 HF transformers 的贪心输出（验证实现正确性）。

权重随机的小模型（如 PrimeIntellect/qwen3-moe-tiny）也能当验证靶：随机权重不影响
力学判据——只要两端在同一 prompt 上给出同样的贪心 token 序列，就说明
RMSNorm / RoPE / GQA / 路由 / MoE / KV cache 组合与 HF 参考实现一致。

用法：
  python scripts/dev/compare_hf.py PrimeIntellect/qwen3-moe-tiny --max-tokens 16
  python scripts/dev/compare_hf.py Qwen/Qwen3-0.6B --kv-bits 8 --max-tokens 24
  python scripts/dev/compare_hf.py Qwen/Qwen3-0.6B --prompt A --prompt B --long-prompt-tokens 600

多 prompt 会**一次性提交**给 zvllm（走 continuous batching / chunked prefill /
prefix cache），HF 侧仍逐条贪心——批处理不改贪心结果，因此这同时是对批不变性的检验。
两侧各自跑在独立子进程：zvllm 会按 gpu_memory_utilization 预占约 90% 显存，
同进程里再加载 HF 模型会超出 11GB 触发换页（表现为 GPU 100% 但长时间无输出）。

贪心逐 token 会放大浮点差异，末端近 tie 处可能分叉。脚本在分叉时用 HF 前向复核
该位置的 logits 排名与 margin：若前两名原始 logit 完全相同，即为 argmax 平局被
不同方式打破（与本仓库 ab_spec_greedy.py 的判据一致），不算实现误差。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
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


def build_prompts(args) -> list[str]:
    prompts = list(args.prompt) or ["The capital of France is"]
    if args.long_prompt_tokens > 0:
        # 长 prompt：重复一段文本直到大致超过指定 token 数，用于触发 chunked prefill
        # 与多块 prefix cache（block_size=256）
        unit = ("In a distant corner of the northern sea there lived an old fisherman "
                "who kept a careful record of every storm he had seen. ")
        repeat = max(1, args.long_prompt_tokens // 18)
        prompts.append(unit * repeat)
    if args.duplicate:
        prompts.append(prompts[0])
    return prompts


def run_child(args: list[str], timeout: int = 1800) -> dict:
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


def side_zvllm(args) -> None:
    from zvllm import LLM, SamplingParams

    prompts = build_prompts(args)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, weight_bits=args.weight_bits,
              kv_bits=args.kv_bits, spec_decode=args.spec, max_model_len=args.max_model_len)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                           use_tqdm=False)
    payload = {
        "ids": [o["token_ids"] for o in outputs],
        "reasons": [o["finish_reason"] for o in outputs],
        "spec_stats": getattr(llm, "spec_stats", None),
    }
    print(f"{MARKER} {json.dumps(payload)}", flush=True)


def side_hf(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from zvllm.utils.model_download import resolve_model_path

    prompts = build_prompts(args)
    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    # dtype 跟随 config（zvllm 侧也是 set_default_dtype(hf_config.dtype)），同 dtype 才可比
    model = AutoModelForCausalLM.from_pretrained(path, dtype="auto").to("cuda").eval()

    ids_all, diags = [], []
    ours_all = json.loads(args.ours_ids) if args.ours_ids else []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens,
                                       use_cache=True, pad_token_id=tokenizer.eos_token_id)
        ids = generated[0][inputs["input_ids"].shape[1]:].tolist()
        ids_all.append(ids)
        ours = ours_all[i] if i < len(ours_all) else []
        diags.append(diagnose(model, tokenizer, prompt, ours, ids))
    print(f"{MARKER} {json.dumps({'ids': ids_all, 'diag': diags})}", flush=True)


def diagnose(model, tokenizer, prompt: str, ours: list[int], ids: list[int]) -> dict:
    """复核分叉位置：返回首个分叉下标与 HF 在该位置的 logit 细节。"""
    import torch

    if not ours or ours == ids:
        return {}
    common = 0
    for a, b in zip(ours, ids):
        if a != b:
            break
        common += 1
    if common >= len(ours):
        return {"common": common}
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    full = torch.cat([inputs["input_ids"], torch.tensor([ours[:common]], device="cuda")], dim=1)
    with torch.inference_mode():
        logits = model(full).logits[0, -1].float()
    top_probs, top_ids = torch.softmax(logits, dim=-1).topk(3)
    chosen = ours[common]
    t1, t2 = int(top_ids[0]), int(top_ids[1])
    return {
        "common": common,
        "chosen": chosen,
        "chosen_rank": int((logits > logits[chosen]).sum().item()) + 1,
        "top1": tokenizer.decode([t1]),
        "top2": tokenizer.decode([t2]),
        "tie": bool(logits[t1] == logits[t2]),
        "margin": float((logits[t1] - logits[t2]).abs().item()),
        "top3_probs": [round(float(p), 4) for p in top_probs.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", action="append", default=None, help="可多次给定")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--long-prompt-tokens", type=int, default=0,
                        help="追加一条约该 token 数的长 prompt（触发 chunked prefill / 多块 prefix cache）")
    parser.add_argument("--duplicate", action="store_true", help="重复第一条 prompt（触发 prefix cache 复用）")
    parser.add_argument("--kv-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--weight-bits", type=int, default=16, choices=[8, 16])
    parser.add_argument("--spec", default="off", choices=["off", "ngram"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--side", choices=["zvllm", "hf"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ours-ids", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.side == "zvllm":
        side_zvllm(args)
        return
    if args.side == "hf":
        side_hf(args)
        return

    common = [args.model, "--max-tokens", str(args.max_tokens), "--kv-bits", str(args.kv_bits),
              "--weight-bits", str(args.weight_bits), "--spec", args.spec,
              "--max-model-len", str(args.max_model_len),
              "--long-prompt-tokens", str(args.long_prompt_tokens)]
    if args.duplicate:
        common.append("--duplicate")
    for p in (args.prompt or []):
        common += ["--prompt", p]

    ours_payload = run_child(["--side", "zvllm", *common, "--gpu-mem", str(args.gpu_mem)])
    theirs_payload = run_child(["--side", "hf", *common, "--ours-ids", json.dumps(ours_payload["ids"])])

    prompts = build_prompts(args)
    tag = (f"kv_bits={args.kv_bits} weight_bits={args.weight_bits} spec={args.spec} "
           f"prompts={len(prompts)} max_tokens={args.max_tokens}")
    print(f"\n===== {tag} =====")
    ok = 0
    for i, prompt in enumerate(prompts):
        ours, theirs = ours_payload["ids"][i], theirs_payload["ids"][i]
        diag = theirs_payload["diag"][i]
        label = f"[{i}] len(prompt)={len(prompt)}"
        if ours == theirs:
            ok += 1
            print(f"{label} 全序列一致（{len(ours)} token）")
            continue
        if diag.get("tie"):
            print(f"{label} 位置 {diag['common']} 分叉，但 HF 前两名 logit 完全相同"
                  f"（{diag['top1']!r} vs {diag['top2']!r}）-> argmax 平局，非实现误差")
        else:
            print(f"{label} 位置 {diag['common']} 分叉：zvllm={diag.get('chosen')} "
                  f"在 HF 中排名第 {diag.get('chosen_rank')}，"
                  f"HF top1={diag.get('top1')!r}/top2={diag.get('top2')!r} margin={diag.get('margin'):.6f}")
    print(f"{ok}/{len(prompts)} 条完全一致；finish_reason={ours_payload['reasons']}；"
          f"spec_stats={ours_payload['spec_stats']}")


if __name__ == "__main__":
    main()
