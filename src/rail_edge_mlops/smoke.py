"""GPU smoke test: prove the container can actually reach the Radeon through ROCm.

`torch.cuda.is_available()` returning True is necessary but not sufficient -- it can
pass while kernels silently produce garbage on an unsupported arch. So this also runs
a real matmul and checks the result against a CPU reference.
"""

from __future__ import annotations

import sys

import torch

EXPECTED_ARCH = "gfx1100"  # Radeon RX 7900 XT (Navi 31)


def main() -> int:
    print(f"torch      : {torch.__version__}")
    print(f"hip        : {torch.version.hip}")

    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible.", file=sys.stderr)
        print(
            "      Check --device=/dev/kfd --device=/dev/dri and video/render groups.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    arch = getattr(props, "gcnArchName", "unknown").split(":")[0]
    print(f"device     : {torch.cuda.get_device_name(device)}")
    print(f"arch       : {arch}")
    print(f"vram       : {props.total_memory / 1024**3:.1f} GB")

    if arch != EXPECTED_ARCH:
        print(f"WARN: expected {EXPECTED_ARCH}, got {arch}.", file=sys.stderr)

    # Real work, not just a capability probe.
    torch.manual_seed(0)
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)
    expected = a @ b
    actual = (a.to(device) @ b.to(device)).cpu()

    max_err = (actual - expected).abs().max().item()
    print(f"matmul err : {max_err:.2e}")

    if not torch.allclose(actual, expected, atol=1e-2, rtol=1e-3):
        print(
            f"FAIL: GPU matmul disagrees with CPU reference (max abs err {max_err:.2e}).",
            file=sys.stderr,
        )
        return 1

    print("OK: GPU reachable and numerically sane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
