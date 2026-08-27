import torch

from rail_edge_mlops import smoke


def test_package_imports() -> None:
    assert callable(smoke.main)


def test_gpu_smoke_passes() -> None:
    """The GPU must be reachable. This is not skipped when absent -- a training
    image that cannot see the Radeon is broken, and CI should say so."""
    assert torch.cuda.is_available(), "no ROCm device visible from inside the container"
    assert smoke.main() == 0
