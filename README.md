# rail-edge-mlops

A closed-loop MLOps pipeline for edge object detection: train on an AMD GPU, deploy to an
NVIDIA Jetson AGX Orin, replay real driving video through it, and detect input/output drift in
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
  nuImages ─► DVC ──► train (PyTorch/ROCm)             CI runner on device
                          │                                  │
                          ├──► MLflow (Postgres+MinIO)        ├──► trtexec ──► TensorRT engine
                          │      experiments + registry       │
                          └──► ONNX + model card ─────────────┤──► parity gate  ─┐
                                 (the handoff contract)       │   ONNX-CPU vs    │ blocks
                                                              │   TRT-FP16/INT8  │ promotion
                                                              ▼                  │
  recorded video ───────► RTSP replay ──────────────► Triton (TensorRT backend) ◄┘
                                                              │
                                                              ▼
                                        Prometheus + Grafana ──► drift monitor
                                        (embedding PSI/MMD, score dist, track churn)
```

## Drift methodology

The drift detector is **label-free by design** — in production there is no ground truth. Labels are
used only offline, to grade the detector.

Evidence comes in tiers, and the distinction matters:

1. **Natural held-out shift (primary evidence).** Data is split *by location*, not randomly:
   train on Singapore, evaluate on Boston. Different continent, architecture, vehicle mix — and
   Singapore drives on the left. Real distribution shift with real labels on both sides, so we can
   show the detector fired *and* accuracy actually dropped. Splitting by location also forces
   grouping by capture log, which prevents near-duplicate frames from the same drive leaking
   across the split and inflating validation scores.
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

- [x] **P0** — reproducible foundation: pinned env, DVC-tracked data, MLflow, split policy
- [ ] **P1** — baseline training on ROCm + evaluation harness
- [ ] **P2** — ONNX handoff contract, on-device TensorRT build, parity gate
- [ ] **P3** — Triton serving + RTSP replay + metrics
- [ ] **P4** — drift detection + induced-drift demonstration
- [ ] **P5** — continuous-training loop; hybrid k3s control plane with the Jetson as an edge node

## Notes on choices

Fuller reasoning, including what each choice gave up: [docs/decisions.md](docs/decisions.md).

**Apache-2.0, and the model follows from it.** The detector is RT-DETRv2 (Apache-2.0) rather than a
YOLO variant, so this repository can stay permissively licensed — Ultralytics' AGPL-3.0 would
otherwise propagate here. RT-DETR was also designed against TensorRT, which suits the deployment
path. The tradeoff: its deformable-attention CUDA kernel does not build on ROCm, so training falls
back to the pure-PyTorch path.

**Why driving data, not rail.** The intended dataset was OSDaR23 (open rail sensor data from
DZSF / Digitale Schiene Deutschland), but its host has been unreachable. nuScenes/nuImages was
chosen instead because its camera + lidar + radar suite matches the same problem shape, and its
metadata carries the location and capture-time attributes the drift split needs. If OSDaR23
returns it becomes an addition rather than a replacement — a detector trained on Boston and
Singapore streets, then fed Hamburg railway footage, is a far stronger drift demonstration than
either dataset alone.

**Public data only.** This project is unaffiliated with, and contains no material from, the
author's employer.

## Data licence and attribution

Training data is [nuImages](https://www.nuscenes.org/nuimages) and
[nuScenes](https://www.nuscenes.org/nuscenes) by **Motional**, used under
**CC BY-NC-SA 4.0** — non-commercial use only. The code in this repository is Apache-2.0; the
data is not, and derived datasets inherit the ShareAlike terms.

## License

[Apache-2.0](LICENSE)
