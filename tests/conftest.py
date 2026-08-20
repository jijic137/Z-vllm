import pytest


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    """共享临时目录（test_models_llama 用它放 HF config / 权重夹具）。"""
    return tmp_path_factory.mktemp("zvllm_test")
