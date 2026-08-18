import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _get_param(model: nn.Module, name: str, skip_unowned: bool):
    try:
        return model.get_parameter(name)
    except AttributeError:
        if skip_unowned:
            return None    # EP 模式下不归属于本 rank 的专家权重
        raise


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    skip_unowned = getattr(model, "skip_unowned_weights", False)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param = _get_param(model, weight_name.replace(k, v), skip_unowned)
                        if param is None:
                            continue
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = _get_param(model, weight_name, skip_unowned)
                    if param is None:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
