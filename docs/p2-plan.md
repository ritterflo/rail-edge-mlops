# P2 — edge deployment and the promotion gate

Train on AMD, deploy on NVIDIA. There is no shared runtime between the two, so
correctness cannot be assumed anywhere in this phase — it has to be measured on the
target device. That constraint is the reason this phase is worth building.

## What P1 changed about this plan

**The parity gate cannot compare at a fixed score threshold.** Three identical training
runs produced maximum scores of 0.375, 0.223 and 0.260 (`reports/threshold_transfer.json`),
so the score distribution moves between models. FP16 rounding moves it again between
runtimes. Comparing "detections above 0.05" between ONNX and TensorRT would report a
regression that is only a rounding difference.

Parity is therefore measured two ways, neither threshold-based:

- **Rank-based** — mAP on both runtimes. Robust to score shifts by construction.
- **Numeric** — maximum absolute divergence in box coordinates and scores, on identical
  inputs. This is the real check: it asks whether the engine computes the same function,
  independent of any threshold.

**The operating threshold is part of the deployable artefact.** Because it does not
transfer between model versions, it cannot live in the serving config as a constant. The
model card carries the threshold derived from that model's own validation split, and the
device applies what the card says.

**The Model Registry is a prerequisite, not a later step.** Artifacts are logged to MinIO
today, but there is no registered model with versions and stages — and "block promotion
on regression" presupposes exactly that.

**No NMS.** RT-DETR's Hungarian matching suppresses duplicates by construction, which
removes the most painful part of exporting a detector to ONNX and running it under
TensorRT. A dividend of the model choice rather than a reason for it.

## Steps

### 0 · Model Registry
Register `rail-edge-detector` in MLflow; training promotes a run's artifact to a new
version. Versions carry the provenance triple already recorded on the run. No GPU.

### 1 · ONNX export and the model card
Export on CPU — ROCm cannot build TensorRT engines, so the training host can never
validate the deployed artefact. That is the constraint this phase exists to handle.

The card is the handoff contract and records: input shape, preprocessing (resize mode,
normalisation), class map, opset, weight sha256, and the operating threshold derived from
this model's validation split.

Input shape comes from `params.yaml:train.image_size`, not a constant — the resolution
experiment may change it to 1024x576, and the shape belongs in configuration regardless.

### 2 · Jetson prerequisites
Flash JetPack 6.2 (L4T 36.4.3) from the Syslogic BSP for the AGX Orin 32GB. Register the
board as a self-hosted runner labelled `jetson`. Verify TensorRT 10.3.

Note the runner hardening already established for the x86 runner applies here too: a
dedicated unprivileged user, and the fork-PR guard in the workflow.

### 3 · On-device engine build
`trtexec`, FP16, built by CI **on the target**. Engines are hardware- and
version-specific; one built anywhere else is not the artefact that will run.

### 4 · Parity gate
On a golden set: mAP delta between ONNX-CPU and TRT-FP16, plus maximum numeric
divergence. Blocks the registry stage transition on regression. This is the deliverable
of the phase.

### 5 · INT8 (optional)
Calibration must run on-device. A separate decision, and one that shifts scores further —
so the threshold must be re-derived for the INT8 artefact rather than inherited from FP16.

## Carried over from P1

- Fit a temperature scaler once training length is settled (recalibrating an
  under-trained model means refitting).
- The resolution experiment (`1024x576`) is unrun; `reports/noise_floor.json` holds the
  thresholds a result must beat.
