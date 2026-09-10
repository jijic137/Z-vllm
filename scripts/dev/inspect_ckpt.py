"""开发用：打印 checkpoint 的权重键与形状（按层前缀过滤），用于核对模型结构与权重命名。

用法：python scripts/dev/inspect_ckpt.py <模型目录|hub id> [层前缀,默认 model.layers.0.]
"""
import glob
import os
import sys

from safetensors import safe_open

from zvllm.utils.model_download import resolve_model_path


def main() -> None:
    model = resolve_model_path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else "model.layers.0."
    files = sorted(glob.glob(os.path.join(model, "*.safetensors")))
    total = 0
    hits = []
    for path in files:
        with safe_open(path, "pt", "cpu") as f:
            for key in f.keys():
                total += 1
                if key.startswith(prefix):
                    hits.append((key, tuple(f.get_slice(key).get_shape()), str(f.get_slice(key).get_dtype())))
    print(f"model: {model}\nsafetensors: {len(files)}  total keys: {total}\nprefix: {prefix}  matched: {len(hits)}")
    for key, shape, dtype in sorted(hits):
        print(f"  {key:<64} {str(shape):<20} {dtype}")


if __name__ == "__main__":
    main()
