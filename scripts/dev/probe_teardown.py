"""探针：引擎销毁后 GPU 显存是否真的释放（诊断"同进程内二次加载模型卡死"）。

背景：一次 HF 对拍脚本卡死（GPU 100%、显存吃满、长时间无输出），怀疑是同一进程里
先建的 zvllm 引擎没释放 KV cache，紧接着加载 HF 模型导致显存超卖。

用法：
  python scripts/dev/probe_teardown.py PrimeIntellect/qwen3-moe-tiny --mode del
  python scripts/dev/probe_teardown.py PrimeIntellect/qwen3-moe-tiny --mode exit
  python scripts/dev/probe_teardown.py PrimeIntellect/qwen3-moe-tiny --mode refs

`--mode sequential` 复现真实场景：同进程里先建一个引擎、丢掉、再建第二个，
验证第二个引擎不会撞上第一个残留的显存（修复前必超卖）。
"""
import argparse
import atexit
import gc
import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402

from zvllm import LLM  # noqa: E402

GIB = 2**30


def mem(tag: str) -> None:
    torch.cuda.synchronize()
    print(f"{tag:<26} allocated={torch.cuda.memory_allocated() / GIB:6.2f} GiB   "
          f"reserved={torch.cuda.memory_reserved() / GIB:6.2f} GiB")


def show_refs(obj) -> None:
    """列出仍强引用该对象的容器（atexit 注册表就是典型的隐形持有者）。"""
    holders = []
    for ref in gc.get_referrers(obj):
        kind = type(ref).__name__
        if kind == "frame":
            continue
        holders.append(kind)
    print(f"  非 frame 引用者：{holders or '（无）'}")
    handlers = getattr(atexit, "_exithandlers", None)
    if handlers is not None:
        bound = [h for h in handlers if getattr(h[0], "__self__", None) is obj]
        print(f"  atexit 注册表里指向本对象的处理器：{len(bound)} 个"
              f"{'（' + bound[0][0].__name__ + '）' if bound else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode", choices=["del", "exit", "refs", "sequential"], default="del")
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    args = parser.parse_args()

    mem("启动")
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem)
    mem("构建引擎后")

    if args.mode == "refs":
        gc.collect()
        show_refs(llm)
        return

    if args.mode == "sequential":
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        mem("丢弃第一个引擎后")
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_mem)
        mem("构建第二个引擎后")
        llm.exit()
        gc.collect()
        torch.cuda.empty_cache()
        mem("退出第二个引擎后")
        return

    if args.mode == "del":
        del llm
    else:
        llm.exit()
    gc.collect()
    torch.cuda.empty_cache()
    mem(f"清理后（{args.mode}）")


if __name__ == "__main__":
    main()
