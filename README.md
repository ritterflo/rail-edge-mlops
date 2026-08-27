# rail-edge-mlops

A closed-loop MLOps pipeline for rail obstacle detection: train on an AMD GPU, deploy to an
NVIDIA Jetson AGX Orin, replay real rail video through it, and detect input/output drift in
operation — with every promotion gated by an on-device acceptance test.

> **Status: early.** Phase 0 (reproducible foundation) in progress. Nothing here is finished yet;
> the commit history is the point as much as the result.

## Why this exists

Most MLOps portfolio projects stop at "train a model, log it to a tracker." The hard parts of
running a model in production are downstream of that: proving the deployed artifact matches the
one you evaluated, and knowing when the world has drifted away from your training distribution
without waiting for labels to tell you.

This project deliberately keeps a constraint most setups engineer away — **training happens on AMD
(ROCm), inference happens on NVIDIA (TensorRT)**. There is no shared runtime between the two, so
correctness cannot be assumed; it has to be *measured on the target device*. That forces an
explicit model handoff contract and a real acceptance gate, which is the piece worth building.

## Architecture

```
  TRAINING HOST (x86, Radeon RX 7900 XT / ROCm)        EDGE (Jetson AGX Orin 32GB / JetPack 6.2)
  ─────────────────────────────────────────────        ──────────────────────────────────────────
  OSDaR23 ──► DVC ──► train (PyTorch/ROCm)             CI runner on device
                          │                                  │
                          ├──► MLflow (Postgres+MinIO)        ├──► trtexec ──► TensorRT engine
                          │      experiments + registry       │
                          └──► ONNX + model card ─────────────┤──► parity gate  ─┐
                                 (the handoff contract)       │   ONNX-CPU vs    │ blocks
                                                              │   TRT-FP16/INT8  │ promotion
                                                              ▼                  │
  recorded rail video ──► RTSP replay ──────────────► Triton (TensorRT backend) ◄┘
                                                              │
                                                              ▼
                                        Prometheus + Grafana ──► drift monitor
                                        (embedding PSI/MMD, score dist, track churn)
```

## Drift methodology

The drift detector is **label-free by design** — in production there is no ground truth. Labels are
used only offline, to grade the detector.

Evidence comes in tiers, and the distinction matters:

1. **Natural held-out shift (primary evidence).** OSDaR23 subsequences are split *by condition*,
   not randomly. Real distribution shift, real labels — so we can show the detector fired *and*
   accuracy actually dropped.
2. **Synthetic corruption sweep (calibration only).** Fog / rain / low-light / motion-blur at
   graded severities. Natural data has no clean severity axis, and thresholds need one. This
   calibrates; it does not prove.
3. **Stated honestly.** Synthetic fog is not fog. Thresholds are *calibrated* on tier 2 and
   *validated* on tier 1.

The headline result is the correlation between the label-free drift score and the offline-measured
accuracy drop.

## Hardware

| Role | Machine |
|---|---|
| Training | Radeon RX 7900 XT (gfx1100, 20 GB), ROCm |
| Edge | Jetson AGX Orin 32 GB, JetPack 6.2 (L4T 36.4.3), Syslogic carrier |

## Roadmap

- [ ] **P0** — reproducible foundation: pinned env, DVC-tracked data, MLflow, split policy
- [ ] **P1** — baseline training on ROCm + evaluation harness
- [ ] **P2** — ONNX handoff contract, on-device TensorRT build, parity gate
- [ ] **P3** — Triton serving + RTSP replay + metrics
- [ ] **P4** — drift detection + induced-drift demonstration
- [ ] **P5** — continuous-training loop; hybrid k3s control plane with the Jetson as an edge node

## Notes on choices

**Apache-2.0, and the model follows from it.** The detector is RT-DETRv2 (Apache-2.0) rather than a
YOLO variant, so this repository can stay permissively licensed — Ultralytics' AGPL-3.0 would
otherwise propagate here. RT-DETR was also designed against TensorRT, which suits the deployment
path. The tradeoff: its deformable-attention CUDA kernel does not build on ROCm, so training falls
back to the pure-PyTorch path.

**Public data only.** OSDaR23 is openly published by DZSF / Digitale Schiene Deutschland. This
project is unaffiliated with, and contains no material from, the author's employer.

## License

[Apache-2.0](LICENSE)
