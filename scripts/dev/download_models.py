"""开发用：从魔搭批量拉取验证用小模型（本地缓存，重复执行不重复下载）。

用法：python scripts/dev/download_models.py Qwen/Qwen3-0.6B ...
不带参数时使用 DEFAULT_MODELS。
"""
import os
import sys

from modelscope import snapshot_download

DEFAULT_MODELS = ["Qwen/Qwen3-0.6B"]


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1024**3


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"cache: {os.environ.get('MODELSCOPE_CACHE', '~/.cache/modelscope')}")
    for model in models:
        path = snapshot_download(model)
        print(f"{model}\n  -> {path}\n  size={dir_size_gb(path):.2f} GB")


if __name__ == "__main__":
    main()
