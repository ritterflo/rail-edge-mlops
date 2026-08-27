# Decisions

Choices made while building this project, with the reasoning and — more usefully — what was
knowingly given up. Anything recorded here was a real fork with a defensible alternative; obvious
calls aren't listed.

Dated 2026-08-27 unless noted.

---

### Apache-2.0, and the detector follows from it

RT-DETRv2 rather than a YOLO variant, so the repository can stay permissively licensed:
Ultralytics is AGPL-3.0 and would propagate here. RT-DETR was also designed against TensorRT,
which suits the deployment path.

*Accepted:* its deformable-attention CUDA kernel doesn't build on ROCm, so training falls back to
the pure-PyTorch path and runs slower. Acceptable at this dataset size on 20 GB of VRAM.

### Public from the first commit

The commit history is part of the demonstration. A clean, well-messaged history from commit one
shows the practice better than a curated reveal would, and it forces the discipline.

*Accepted:* the early mistakes are visible — including a broken `.gitignore` and a CI failure
caused by running containers as root. That's the honest version.

### Training runs in a container, not on the host

ROCm 7.2.1 supports Ubuntu 24.04 and 22.04; this workstation runs 26.04, which is outside AMD's
support matrix. A pinned `rocm/dev-ubuntu-24.04` userspace over the host's amdgpu/KFD kernel
driver puts the toolchain back inside a supported configuration, and insulates it from host
library churn.

*Accepted:* GPU passthrough plumbing, and a ~25 GB image.

### The image tag is a content hash of its inputs

`sha256(docker/** + uv.lock)`, not the commit SHA. An unchanged input means the tag already
exists and nothing rebuilds; a changed pin produces a tag that doesn't exist yet, which *is* the
rebuild trigger. Tagging by commit would rebuild a 25 GB image on every push.

### No image registry

The training image (x86/ROCm) and the future edge image (arm64/JetPack) live on different
machines and never need to meet, and the only consumer of each is the machine that built it.
Pushing 25 GB from a home connection buys nothing. Reproducibility comes from the committed
recipe, not a hosted binary.

*Accepted:* nobody else can pull the exact image; they rebuild from the Dockerfile. If a
pullable artifact is ever wanted for demonstration, it should be the slim inference image.

### Containers run as the invoking user

Root-owned pytest and ruff caches written into the bind-mounted workspace made CI's next
`git clean -ffdx` fail with `EACCES`. Every push after the first would have broken regardless of
contents. Group IDs are the host's numeric ones because `--group-add render` resolves the name
inside the container, where `render` is 109 while `/dev/kfd` is owned by the host's 110 —
running as root had masked the mismatch entirely.

### CI is hermetic; integration is a separate lane

`make lint/test/smoke` need no running services, so a red build means the code is wrong rather
than that a daemon was restarting. `make check-services` and the data pipeline need the real
stack and are run deliberately. This holds even though the services are always up — a test suite
that fails during an unrelated upgrade trains you to ignore red.

### Services are long-lived; their state lives outside the repo

`restart: unless-stopped` plus Docker's boot unit means Postgres, MinIO and MLflow behave like
infrastructure rather than something to start each session. Their volumes sit under an absolute
`DATA_ROOT` outside the working tree, so renaming or re-cloning the repo cannot orphan the
databases, and tens of GB of data never make repo size meaningless.

### Data is imported by URL, not added locally

`dvc import-url` records the source URL and etag alongside the content hash, so the `.dvc` file
answers *where did this come from* and `dvc update` can detect upstream changes. `dvc add` would
record only *what* it is.

### nuImages instead of OSDaR23

The intended dataset was OSDaR23 (open rail sensor data, DZSF), but its host has been
unreachable and there is no mirror. nuScenes/nuImages has a camera + lidar + radar suite
matching the same problem shape, carries the location and capture-time metadata the split policy
needs, and is mirrored on public S3.

*Accepted:* CC BY-NC-SA 4.0, non-commercial, and it isn't rail. If OSDaR23 returns it becomes an
addition rather than a replacement — a detector trained on Boston and Singapore streets then fed
Hamburg railway footage is a stronger drift demonstration than either dataset alone.

### The nuScenes 10-class detection taxonomy

nuImages' 25 fine-grained categories collapse to the published nuScenes 10. Using the standard
taxonomy makes the resulting mAP a number someone else can check, rather than one only this
repository can produce.

*Accepted:* seven boxes in the mini split fall outside the mapping (animal, debris, stroller,
bicycle rack). The converter counts and prints every dropped annotation, so "dropped" can't
become "silently lost".

### Split by location, grouped by capture log, three ways

The test set is an entire held-out location — train on Singapore, test on Boston. Different
continent, architecture and vehicle mix, and Singapore drives on the left, so it is a genuine
distribution shift rather than a reshuffle. Every boundary falls between logs, never inside one:
frames from the same drive are seconds apart and near-identical, and splitting at image level
would put near-duplicates on both sides and inflate validation until the model met real data.
Validation is carved from the training locations so the test location is never tuned against.

### Upstream drift and kernel nondeterminism are accepted, not fought

The base image is pinned by tag rather than digest, apt packages float, and GPU kernels are left
nondeterministic. Exact apt pinning is self-defeating — Ubuntu's archive keeps only the current
version per pocket, so a precisely pinned Dockerfile eventually fails to build at all. Forcing
deterministic kernels costs throughput.

The underlying judgement: a model that only reaches its numbers under one exact kernel schedule
has a robustness problem, not a reproducibility problem.

*Instead:* run-to-run variance is measured and reported, so every later comparison is read
against a known noise floor. A 1.5-point gain means something different at ±0.4 than at ±2.5,
and the drift results in P4 are only falsifiable if the noise floor is known.

*Escape hatch:* for any model that gets deployed and quoted, `docker save` the image and store it
beside the model. Recipes rot; tarballs don't.
