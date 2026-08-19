"""模型权重解析与下载：本地路径 / 魔搭（ModelScope）/ Hugging Face。

model_source 取值：
- "auto"（默认）：已是本地目录则直接用；否则先尝试 modelscope（国内网络更稳定），
  失败再降级 huggingface
- "modelscope"：强制走魔搭社区下载
- "hf"：强制走 Hugging Face 下载
"""
import os


def _download_modelscope(model: str) -> str:
    from modelscope import snapshot_download
    return snapshot_download(model)


def _download_hf(model: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


def resolve_model_path(model: str, source: str = "auto") -> str:
    """把模型标识解析成本地目录。

    model 为已存在的本地目录时原样返回（不触发任何下载）；
    否则按 source 从 hub 下载（snapshot 缓存在各 hub 的默认目录，重复调用不重复下载）。
    """
    if os.path.isdir(model):
        return model
    if source == "modelscope":
        return _download_modelscope(model)
    if source == "hf":
        return _download_hf(model)
    if source == "auto":
        errors = []
        for fn in (_download_modelscope, _download_hf):
            try:
                return fn(model)
            except Exception as e:    # 依赖未安装 / 网络不通 / hub 上不存在该模型
                errors.append(f"  {fn.__name__}: {type(e).__name__}: {e}")
        raise RuntimeError(f"模型 {model} 下载失败：\n" + "\n".join(errors))
    raise ValueError(f"未知 model_source: {source}（可选 auto / modelscope / hf）")
