"""DP（多副本）vs 单引擎推理基准：同一负载下测 TTFT / TPOT / 吞吐。

指标：
  - TTFT：每请求首 token 延迟（mean / median / p95，ms）
  - TPOT：每请求逐 token 平均延迟（mean / median，ms）
  - 吞吐：总产出 token / 全程 wall time（tok/s）
  - 并发：C 个 worker 线程同时持有请求（每线程跑完一个请求再取下一个）

单引擎（--dp 1）：LLMEngine + 独立 step 线程（与 api_server 相同的
schedule → run_model → finalize_step 三段驱动）。
DP（--dp >1）：DPClient 多副本客户端（每副本独立进程组，父进程路由 + 驱动）。

运行（服务器 4 卡示例，两组数字同负载可比）：
  CUDA_VISIBLE_DEVICES=2,3,4,5 python bench_dp.py --model Qwen/Qwen3-30B-A3B \
      --dp 1 --tp 4 --ep 4 --gpus 2,3,4,5 --concurrency 16
  CUDA_VISIBLE_DEVICES=2,3,4,5 python bench_dp.py --model Qwen/Qwen3-30B-A3B \
      --dp 2 --tp 2 --ep 2 --gpus 2,3,4,5 --concurrency 16
每次运行输出一行 DP_BENCH: JSON，便于多并发点汇总。
"""
import argparse
import json
import statistics as st
import threading
import time
from random import randint, seed

from zvllm.sampling_params import SamplingParams


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class _SingleEngineBackend:
    """单引擎：LLMEngine + step 线程（与 api_server._step_loop 同构，去掉锁与推送）。"""

    def __init__(self, model, tp, master_port, **engine_kwargs):
        from zvllm.engine.llm_engine import LLMEngine
        self.engine = LLMEngine(model, tensor_parallel_size=tp,
                                master_port=master_port, **engine_kwargs)
        self._stop = threading.Event()
        threading.Thread(target=self._step_loop, name="bench-step", daemon=True).start()

    def _step_loop(self):
        engine = self.engine
        while not self._stop.is_set():
            if engine.scheduler.is_finished():
                time.sleep(0.005)
                continue
            seqs, is_prefill = engine.schedule()
            if not seqs:
                time.sleep(0.005)
                continue
            token_ids = engine.run_model(seqs, is_prefill)
            engine.finalize_step(seqs, is_prefill, token_ids)

    def run(self, prompt, sp):
        """跑完一个请求，返回 (ttft_s, t_end_s, n_tok)。轮询序列状态（1ms 粒度）。"""
        t0 = time.perf_counter()
        seq = self.engine.add_request(prompt, sp)
        ttft = None
        n = 0
        while not seq.is_finished:
            time.sleep(0.001)
            cur = len(seq.completion_token_ids)
            if cur > n:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n = cur
        if ttft is None:
            ttft = time.perf_counter() - t0
        return ttft, time.perf_counter() - t0, n

    def close(self):
        self._stop.set()
        self.engine.exit()


class _DPBackend:
    """DP 多副本：DPClient（add_request + stream 事件流，无轮询抖动）。"""

    def __init__(self, model, dp, tp, gpus, master_port, **engine_kwargs):
        from zvllm.engine.dp_engine import DPClient
        self.client = DPClient(model, dp_size=dp, tensor_parallel_size=tp, gpus=gpus,
                               master_port=master_port, **engine_kwargs)

    def run(self, prompt, sp):
        t0 = time.perf_counter()
        req_id = self.client.add_request(prompt, sp)
        ttft = None
        n = 0
        for toks, finished, reason in self.client.stream(req_id):
            if ttft is None and toks:
                ttft = time.perf_counter() - t0
            n += len(toks)
            if finished:
                break
        if ttft is None:
            ttft = time.perf_counter() - t0
        return ttft, time.perf_counter() - t0, n

    def close(self):
        self.client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--gpus", default="", help="全局 GPU 号列表，如 2,3,4,5（DP 必填）")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--master-port", type=int, default=2345)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x] or None
    engine_kwargs = dict(max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         moe_ep_size=args.ep,
                         enforce_eager=args.enforce_eager)

    if args.dp > 1:
        assert gpus and len(gpus) == args.dp * args.tp, \
            f"--gpus 数量（{len(gpus) if gpus else 0}）需等于 dp×tp（{args.dp * args.tp}）"
        backend = _DPBackend(args.model, args.dp, args.tp, gpus, args.master_port, **engine_kwargs)
        tag = f"dp{args.dp}xtp{args.tp}xep{args.ep}"
    else:
        backend = _SingleEngineBackend(args.model, args.tp, args.master_port, **engine_kwargs)
        tag = f"tp{args.tp}xep{args.ep}"

    seed(0)
    prompts = [[randint(0, 10000) for _ in range(args.prompt_len)]
               for _ in range(args.num_prompts)]
    sp = SamplingParams(temperature=0, max_tokens=args.gen_len, ignore_eos=True)

    print(f"=== bench_dp {tag} C={args.concurrency} N={args.num_prompts} "
          f"pl={args.prompt_len} gl={args.gen_len} ===", flush=True)

    # warmup（排除首步编译/alloc 噪声）
    backend.run(prompts[0][:64], SamplingParams(temperature=0, max_tokens=16))

    results = [None] * args.num_prompts
    lock = threading.Lock()

    def worker(idx):
        r = backend.run(prompts[idx], sp)
        with lock:
            results[idx] = r

    t0 = time.perf_counter()
    for start in range(0, args.num_prompts, args.concurrency):
        batch = range(start, min(start + args.concurrency, args.num_prompts))
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in results]
    tpots = [(e - f) / (n - 1) for f, e, n in results if n > 1]
    total_tokens = sum(r[2] for r in results)
    out = {
        "tag": tag, "dp": args.dp, "tp": args.tp, "ep": args.ep,
        "concurrency": args.concurrency, "num_prompts": args.num_prompts,
        "prompt_len": args.prompt_len, "gen_len": args.gen_len,
        "ttft_mean_ms": st.mean(ttfts) * 1e3,
        "ttft_med_ms": st.median(ttfts) * 1e3,
        "ttft_p95_ms": pct(ttfts, 95) * 1e3,
        "tpot_mean_ms": st.mean(tpots) * 1e3,
        "tpot_med_ms": st.median(tpots) * 1e3,
        "total_tokens": total_tokens,
        "wall_s": wall,
        "tps": total_tokens / wall,
        "per_req": [[round(f * 1e3, 1), round(e * 1e3, 1), n] for f, e, n in results],
    }
    print("DP_BENCH: " + json.dumps(out), flush=True)
    print(f"TTFT mean/med/p95 = {out['ttft_mean_ms']:.1f}/{out['ttft_med_ms']:.1f}/"
          f"{out['ttft_p95_ms']:.1f} ms")
    print(f"TPOT mean/med     = {out['tpot_mean_ms']:.2f}/{out['tpot_med_ms']:.2f} ms")
    print(f"吞吐 = {out['tps']:.1f} tok/s（{total_tokens} tok / {wall:.2f}s）")

    backend.close()


if __name__ == "__main__":
    main()
