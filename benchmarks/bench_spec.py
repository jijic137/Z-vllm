"""投机解码性能基准：spec off vs on（同一负载，进程级隔离各跑一遍）。

指标：
  - TTFT：每 prompt 首 token 延迟（mean / median / p95，ms）
  - TPOT：每 prompt 逐 token 平均延迟（mean / median，ms）
  - 吞吐：总产出 token / 全程 wall time（tok/s）
  - on 侧额外：接受率、spec 步数、proposed/accepted

运行（服务器单卡）：CUDA_VISIBLE_DEVICES=4 python bench_spec.py
"""
import json
import statistics as st
import subprocess
import sys
import time
from random import randint, seed

MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
N_SEQS = 8
MAXTOK = 192
PROMPT_LENS = [128, 160, 200, 256, 128, 300, 160, 200]


def child(mode: str):
    from zvllm.engine.llm_engine import LLMEngine
    from zvllm.sampling_params import SamplingParams

    seed(0)
    prompts = [[randint(0, 10000) for _ in range(n)] for n in PROMPT_LENS]
    engine = LLMEngine(
        MODEL, tensor_parallel_size=1, spec_decode=mode, spec_gamma=4, spec_ngram=4,
        max_model_len=2048, gpu_memory_utilization=0.8, master_port=2345,
    )
    # warmup（排除首步编译/alloc 噪声）
    engine.generate([prompts[0][:32]], [SamplingParams(temperature=0, max_tokens=16)], use_tqdm=False)

    t0 = time.perf_counter()
    ttft = [None] * N_SEQS
    t_end = [None] * N_SEQS
    n_tok = [0] * N_SEQS
    gen = engine.generate(
        prompts,
        [SamplingParams(temperature=0, max_tokens=MAXTOK, ignore_eos=True)] * N_SEQS,
        use_tqdm=False, stream=True,
    )
    for ev in gen:
        i = ev["index"]
        n_tok[i] = len(ev["token_ids"])
        if ttft[i] is None:
            ttft[i] = time.perf_counter() - t0
        if ev["finished"]:
            t_end[i] = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    stats = dict(engine.spec_stats)
    engine.exit()
    print("BENCH_RESULT " + json.dumps({
        "mode": mode, "ttft_s": ttft, "t_end_s": t_end, "n_tok": n_tok,
        "wall_s": wall, "total_tokens": sum(n_tok), "spec_stats": stats,
    }))


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(r):
    tpots = [(t - f) / (n - 1) for t, f, n in zip(r["t_end_s"], r["ttft_s"], r["n_tok"]) if n > 1]
    return {
        "ttft_mean_ms": st.mean(r["ttft_s"]) * 1e3,
        "ttft_p95_ms": pct(r["ttft_s"], 95) * 1e3,
        "tpot_mean_ms": st.mean(tpots) * 1e3,
        "tpot_med_ms": st.median(tpots) * 1e3,
        "tps": r["total_tokens"] / r["wall_s"],
        "total_tokens": r["total_tokens"],
        "wall_s": r["wall_s"],
        "spec_stats": r["spec_stats"],
    }


def main():
    results = {}
    for mode in ("off", "ngram"):
        print(f"run mode={mode} ...", flush=True)
        p = subprocess.run([sys.executable, sys.argv[0], mode], capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            sys.exit(1)
        lines = [l for l in p.stdout.splitlines() if l.startswith("BENCH_RESULT ")]
        results[mode] = json.loads(lines[-1][len("BENCH_RESULT "):])

    so, sn = summarize(results["off"]), summarize(results["ngram"])
    acc = sn["spec_stats"]
    acc_rate = acc.get("accepted_tokens", 0) / max(1, acc.get("proposed_tokens", 1))
    print("=== bench_spec (Qwen3-0.6B, 单卡, bf16, N=8 x 192tok, 贪心) ===")
    print(f"{'指标':<16}{'spec off':>14}{'spec on':>14}{'加速比':>10}")
    print(f"{'TTFT mean(ms)':<16}{so['ttft_mean_ms']:>14.1f}{sn['ttft_mean_ms']:>14.1f}{so['ttft_mean_ms']/max(sn['ttft_mean_ms'],1e-9):>10.2f}")
    print(f"{'TTFT p95(ms)':<16}{so['ttft_p95_ms']:>14.1f}{sn['ttft_p95_ms']:>14.1f}{so['ttft_p95_ms']/max(sn['ttft_p95_ms'],1e-9):>10.2f}")
    print(f"{'TPOT mean(ms)':<16}{so['tpot_mean_ms']:>14.2f}{sn['tpot_mean_ms']:>14.2f}{so['tpot_mean_ms']/max(sn['tpot_mean_ms'],1e-9):>10.2f}")
    print(f"{'TPOT med(ms)':<16}{so['tpot_med_ms']:>14.2f}{sn['tpot_med_ms']:>14.2f}{so['tpot_med_ms']/max(sn['tpot_med_ms'],1e-9):>10.2f}")
    print(f"{'吞吐(tok/s)':<16}{so['tps']:>14.1f}{sn['tps']:>14.1f}{sn['tps']/max(so['tps'],1e-9):>10.2f}")
    print(f"总 token: off={so['total_tokens']} on={sn['total_tokens']}  "
          f"wall: off={so['wall_s']:.2f}s on={sn['wall_s']:.2f}s")
    print(f"spec on: {acc}  accept_rate={acc_rate:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()
