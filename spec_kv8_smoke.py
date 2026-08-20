"""Smoke：spec(ngram) off vs on 在 kv8 下单流 256 token，期望逐位一致（验证 varlen int8 路径）。

判据（同 ab_spec_greedy 的 A 门槛）：单序列贪心，spec on 与 off 输出 token 流逐位一致。
on 侧每步验证 1+gamma 个 token，走 varlen attention 读 int8 cache（此前上下文）+
新 bf16 K/V（草稿 token），与纯 decode 的 int8 路径共用同一 store kernel。

运行（服务器，单卡）：HIP_VISIBLE_DEVICES=4 ~/zvllm-env/bin/python spec_kv8_smoke.py
"""
import torch

from zvllm import LLM, SamplingParams

MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
PROMPT = "Explain step by step what a block table does in a paged KV cache system."
MAXTOK = 256


def run(spec: str, port: int, shm: str):
    llm = LLM(MODEL, enforce_eager=True, kv_bits=8, max_model_len=1024,
              spec_decode=spec, master_port=port, shm_name=shm)
    sp = SamplingParams(temperature=0, max_tokens=MAXTOK, ignore_eos=True)
    outs = llm.generate([PROMPT], sp, use_tqdm=False)
    ids = list(outs[0]["token_ids"])
    text = outs[0]["text"]
    llm.exit()
    torch.cuda.empty_cache()
    return ids, text


def main():
    torch.manual_seed(0)
    off_ids, off_text = run("off", 2361, "zkvsma")
    on_ids, on_text = run("ngram", 2362, "zkvsmb")
    same = off_ids == on_ids
    print(f"[spec_kv8_smoke] off len={len(off_ids)}  on len={len(on_ids)}  "
          f"bit_identical={same}", flush=True)
    if not same:
        for i, (a, b) in enumerate(zip(off_ids, on_ids)):
            if a != b:
                print(f"  first_diverge at {i}: off={a} on={b}", flush=True)
                break
    print("  [off]", off_text[:220], flush=True)
    print("  [on] ", on_text[:220], flush=True)
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
