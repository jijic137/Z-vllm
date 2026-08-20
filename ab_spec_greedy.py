"""GPU A/B：投机解码贪心一致性门槛（spec on vs off，近 tie 感知）。

判据（Qwen3-0.6B，单卡，bf16，贪心）：
  A（硬）：off1 vs on1（单序列）输出 token 流逐位一致；
  B（容差）：off3 vs on3（3 序列）逐 prompt：要么逐位一致，要么 spec 侧用过草稿时
     首个分歧点的两侧决定行满足 min(gap_top2) <= NEAR_TIE（bf16 近 tie），且分歧点
     之前两侧决定行 argmax 逐位一致（无早翻）；
  C（硬）：on3b（另起进程重跑）与 on3 逐位一致（跨进程确定性）。

为什么 B 用容差：多序列批处理时验证步的 logit 行形状（每序列 1+γ 行）与纯 decode
步（每序列 1 行）不同，SDPA/bmm 的归约顺序跨 batch 形状不保证逐位一致，logit 带
几个 ulp 的 bf16 漂移，top-2 近 tie 时 argmax 会翻转。这是 bf16 的固有数值行为
而非 bug（根因证据链见 README"投机解码"节的 B2 实验）；单序列路径逐位一致
（verify M=5 vs decode M=1 实测 3/3），故 A 仍为硬门槛。

运行（服务器，单卡）：CUDA_VISIBLE_DEVICES=4 python ab_spec_greedy.py

父进程派生 5 个独立子进程（进程级隔离：同进程顺序构建引擎时，前引擎的 caching
allocator 持有已释放的 KV 池，后引擎 KV 预算算成负数 assert）。每个子进程逐步
记录每序列 (L, g, out, 各行 top-2/top-5)，落盘 /tmp/abg_<mode>.json。
"""
import json
import subprocess
import sys

MODEL = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
PROMPTS = [
    "Write a short poem about the sea.",
    "Explain step by step what a block table does in a paged KV cache system.",
    "def fib(n):\n    \"\"\"compute the n-th fibonacci number\"\"\"\n",
]
MAXTOK = 128
NEAR_TIE = 0.5
MODES = ("off1", "on1", "off3", "on3", "on3b")


def child(mode: str):
    spec = "off" if mode.startswith("off") else "ngram"
    prompts = PROMPTS[:1] if mode.endswith("1") else PROMPTS

    from zvllm.engine.llm_engine import LLMEngine
    from zvllm.engine.model_runner import ModelRunner
    from zvllm.sampling_params import SamplingParams

    state = {"active": False}
    entries: list[dict] = []
    seq2idx: dict[int, int] = {}

    def row_stats(logits, row0, nrows):
        rows = []
        for r in range(row0, row0 + nrows):
            top5 = logits[r].float().topk(5)
            idx = [int(x) for x in top5.indices]
            val = [round(float(x), 6) for x in top5.values]
            # [top1_idx, top1_val, gap_top2, top5 列表]
            rows.append([idx[0], val[0], round(val[0] - val[1], 6), list(zip(idx, val))])
        return rows

    orig_run_model = ModelRunner.run_model

    def patched_run_model(self, input_ids, positions, is_prefill):
        logits = orig_run_model(self, input_ids, positions, is_prefill)
        if state["active"]:
            self._dbg_logits = logits
        return logits

    orig_run = ModelRunner.run

    def patched_run(self, seqs, is_prefill):
        token_lists = orig_run(self, seqs, is_prefill)
        if state["active"] and token_lists is not None:
            logits = getattr(self, "_dbg_logits", None)
            if logits is not None:
                # 行布局与 _sample_all 严格并行：非投机序列 1 行，投机序列 1+g 行
                row = 0
                for i, seq in enumerate(seqs):
                    g = len(seq.draft_tokens)
                    nrows = g + 1 if g else 1
                    entries.append({
                        "i": seq2idx.get(seq.seq_id, -1),
                        "L": len(seq),          # 本步前已提交长度（草稿未提交）
                        "g": g,                 # 本草稿数（0 = 纯 decode 步）
                        "out": [int(t) for t in token_lists[i]],
                        "rows": row_stats(logits, row, nrows),
                    })
                    row += nrows
        return token_lists

    orig_add = LLMEngine._add_requests

    def patched_add(self, prompts_, sampling_params):
        seqs = orig_add(self, prompts_, sampling_params)
        for i, s in enumerate(seqs):
            seq2idx[s.seq_id] = i
        return seqs

    ModelRunner.run_model = patched_run_model
    ModelRunner.run = patched_run
    LLMEngine._add_requests = patched_add

    engine = LLMEngine(
        MODEL,
        tensor_parallel_size=1,
        spec_decode=spec,
        spec_gamma=4,
        spec_ngram=4,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        master_port=2345,
    )
    state["active"] = True
    try:
        outs = engine.generate(
            prompts,
            [SamplingParams(temperature=0, max_tokens=MAXTOK)] * len(prompts),
            use_tqdm=False,
        )
    finally:
        state["active"] = False
    stats = dict(engine.spec_stats)
    engine.exit()
    result = {"mode": mode,
              "token_ids": [o["token_ids"] for o in outs],
              "entries": entries,
              "stats": stats}
    with open(f"/tmp/abg_{mode}.json", "w") as f:
        json.dump(result, f)
    print("ABG_RESULT " + json.dumps(result), flush=True)


def first_diff(a, b):
    m = min(len(a), len(b))
    for j in range(m):
        if a[j] != b[j]:
            return j
    # 长度不等时返回公共前缀长度（第一个缺失 token 的位置）
    return m if len(a) != len(b) else None


def timeline(res):
    """按 prompt 切分子进程记录：tl[i] = 该 prompt 的逐步记录（按步序）。"""
    n = max([e["i"] for e in res["entries"]] + [-1]) + 1
    tls = [[] for _ in range(n)]
    for e in res["entries"]:
        if e["i"] >= 0:
            tls[e["i"]].append(e)
    return tls


def deciding_row(tl, pos):
    """第 pos 个（0 起）产出 token 的决定行：行号 = pos - c（c 为本步前已产出数），
    纯 decode 步恒为第 0 行。返回行记录 [top1_idx, top1_val, gap_top2, top5]。"""
    c = 0
    for e in tl:
        if c <= pos < c + len(e["out"]):
            k = pos - c
            if e["g"] == 0:
                assert k == 0, f"position {pos} 不在非投机步首个产出"
                return e["rows"][0]
            assert k <= e["g"], f"position {pos} 超出本步 {e['g'] + 1} 行"
            return e["rows"][k]
        c += len(e["out"])
    raise AssertionError(f"position {pos} 超出产出范围")


def fmt_top5(top5):
    return " ".join(f"{i}:{v:.3f}" for i, v in top5)


def check_a(res_off, res_on):
    a, b = res_off["token_ids"][0], res_on["token_ids"][0]
    d = first_diff(a, b)
    ok = d is None
    if ok:
        print(f"[A] off1 vs on1: 逐位一致（{len(a)} tokens）PASS")
    else:
        print(f"[A] off1 vs on1: FAIL 首个分歧 at {d}（off={a[d]} on={b[d]}）")
    return ok


def check_b(i, tl_off, tl_on, tids_off, tids_on):
    ever_draft = any(e["g"] > 0 for e in tl_on)
    d = first_diff(tids_off, tids_on)
    if d is None:
        print(f"[B] prompt {i}: 逐位一致（{len(tids_off)} tokens, draft_used={ever_draft}）PASS")
        return True
    if d >= min(len(tids_off), len(tids_on)):
        # 长度型分歧：公共前缀逐位一致、仅短侧提前停止。贪心下终止是前缀的
        # 函数（EOS/停止串/max_tokens），两侧同前缀必同终止，故不可解释 -> FAIL
        print(f"[B] prompt {i}: FAIL 长度型分歧（{len(tids_off)} vs {len(tids_on)} tokens，"
              f"公共前缀一致但短侧提前停止，贪心下不可解释）")
        return False
    if not ever_draft:
        print(f"[B] prompt {i}: FAIL 首个分歧 at {d}，但 spec 侧从未用过草稿（应逐位一致）")
        return False
    row_off = deciding_row(tl_off, d)
    row_on = deciding_row(tl_on, d)
    # 自洽：贪心输出必等于决定行 argmax（同时校验行映射记账）
    if row_off[0] != tids_off[d] or row_on[0] != tids_on[d]:
        print(f"[B] prompt {i}: FAIL 行映射不自洽"
              f"（off argmax={row_off[0]} vs out={tids_off[d]}，"
              f"on argmax={row_on[0]} vs out={tids_on[d]}）")
        return False
    for p in range(d):
        ro, rn = deciding_row(tl_off, p), deciding_row(tl_on, p)
        if ro[0] != rn[0]:
            print(f"[B] prompt {i}: FAIL 分歧点之前 argmax 早翻 at {p}（off={ro[0]} on={rn[0]}）")
            return False
    min_gap = min(row_off[2], row_on[2])
    ok = min_gap <= NEAR_TIE
    print(f"[B] prompt {i}: 首个分歧 at {d}（off={tids_off[d]} on={tids_on[d]}），此前逐位一致")
    print(f"      off 决定行: top1={row_off[0]}({row_off[1]:.4f}) gap={row_off[2]:.4f} "
          f"top5={fmt_top5(row_off[3])}")
    print(f"      on  决定行: top1={row_on[0]}({row_on[1]:.4f}) gap={row_on[2]:.4f} "
          f"top5={fmt_top5(row_on[3])}")
    print(f"      min_gap={min_gap:.4f} vs NEAR_TIE={NEAR_TIE} -> "
          f"{'NEAR-TIE PASS' if ok else 'NOT NEAR-TIE FAIL'}")
    return ok


def check_c(res_a, res_b):
    ok = True
    for i, (a, b) in enumerate(zip(res_a["token_ids"], res_b["token_ids"])):
        same = a == b
        ok = ok and same
        print(f"[C] prompt {i}: on3 vs on3b identical={same}（{len(a)} tokens）")
    return ok


def main():
    results = {}
    for mode in MODES:
        print(f"run mode={mode} ...", flush=True)
        p = subprocess.run([sys.executable, sys.argv[0], mode],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            sys.exit(1)
        lines = [l for l in p.stdout.splitlines() if l.startswith("ABG_RESULT ")]
        if not lines:
            print("child produced no ABG_RESULT line")
            print(p.stdout[-2000:])
            print(p.stderr[-2000:])
            sys.exit(1)
        results[mode] = json.loads(lines[-1][len("ABG_RESULT "):])

    all_ok = check_a(results["off1"], results["on1"])
    tl_off3, tl_on3 = timeline(results["off3"]), timeline(results["on3"])
    for i in range(len(PROMPTS)):
        all_ok &= check_b(i, tl_off3[i], tl_on3[i],
                          results["off3"]["token_ids"][i],
                          results["on3"]["token_ids"][i])
    all_ok &= check_c(results["on3"], results["on3b"])

    st = results["on3"]["stats"]
    rate = f"{st['accepted_tokens'] / st['proposed_tokens']:.3f}" if st.get("proposed_tokens") else "n/a"
    print(f"spec_stats(on3): {st} accept_rate={rate}")
    print("ABG_PASS" if all_ok else "ABG_FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()
