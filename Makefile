# Single entrypoint for every containerised task. CI calls these same targets,
# so "works locally" and "works in CI" cannot diverge.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# The tag IS the content hash of the image's inputs. Change docker/** or uv.lock
# and the tag changes, no image with that tag exists, and `image` rebuilds.
# `find | sort` because find's order is not deterministic.
TAG   := $(shell { find docker -type f | sort | xargs cat; cat uv.lock; } | sha256sum | cut -c1-12)
IMAGE := rail-edge/train:$(TAG)

# --device kfd/dri + video/render groups: ROCm GPU passthrough.
# --ipc=host --shm-size: PyTorch dataloader workers die on the default 64MB /dev/shm.
# Run as the invoking user, not root: anything the container writes into the
# bind-mounted workspace (caches, checkpoints, exports) would otherwise be
# root-owned, and CI's next `git clean -ffdx` cannot delete those files.
HOST_UID   := $(shell id -u)
HOST_GID   := $(shell id -g)
# The image has a passwd entry for uid 1000 (Ubuntu's `ubuntu` user) and nothing else,
# so DVC's getpwuid() succeeds by luck for uid 1000 and dies for any other -- including
# the CI runner's 1001. getpass.getuser() checks USER before falling back to the passwd
# database, so exporting it makes the lookup work for any uid.
HOST_USER  := $(shell id -un)
# Numeric HOST GIDs, deliberately. `--group-add render` resolves the name against
# the CONTAINER's /etc/group, which maps it to a different number (109 vs the
# host's 110). As root that went unnoticed; as a normal user the GID must match
# the device file's real owner or /dev/kfd is unreadable.
VIDEO_GID  := $(shell getent group video  | cut -d: -f3)
RENDER_GID := $(shell getent group render | cut -d: -f3)

# nofile: dataloader workers hold many descriptors; the default 1024 is exhausted
# thousands of batches into a run, which is an expensive place to discover it.
GPU_ARGS := --user $(HOST_UID):$(HOST_GID) -e HOME=/tmp -e USER=$(HOST_USER) \
	--ulimit nofile=65536:65536 \
	--device=/dev/kfd --device=/dev/dri \
	--group-add $(VIDEO_GID) --group-add $(RENDER_GID) \
	--security-opt seccomp=unconfined \
	--ipc=host --shm-size=8g

DOCKER_RUN := docker run --rm $(GPU_ARGS) \
	-v $(PWD):/workspace -w /workspace \
	$(IMAGE)

.PHONY: help image shell lint test smoke tag clean-image

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

tag:  ## Print the resolved image tag
	@echo $(IMAGE)

image:  ## Build the training image if this content hash has no image yet
	@docker image inspect $(IMAGE) >/dev/null 2>&1 \
		&& echo "$(IMAGE) already built" \
		|| docker build -f docker/Dockerfile.train -t $(IMAGE) .

shell: image  ## Interactive shell inside the training image
	@docker run --rm -it $(GPU_ARGS) -v $(PWD):/workspace -w /workspace $(IMAGE) /bin/bash

lint: image  ## Ruff lint + format check
	$(DOCKER_RUN) ruff check .
	$(DOCKER_RUN) ruff format --check .

test: image  ## Run the test suite
	$(DOCKER_RUN) pytest -q

smoke: image  ## Prove the GPU is really reachable from inside the container
	$(DOCKER_RUN) python3 -m rail_edge_mlops.smoke

clean-image:  ## Remove images for older content hashes
	@docker images 'rail-edge/train' --format '{{.Repository}}:{{.Tag}}' \
		| grep -v '$(TAG)' | xargs -r docker rmi

# --- Local services (Postgres + MinIO + MLflow) --------------------------
# `restart: unless-stopped` means these are long-lived infrastructure: bring
# them up once and Docker restores them across reboots.
COMPOSE := docker compose --env-file .env -f infra/compose.yaml

.PHONY: services-up services-down services-ps services-logs

services-up:  ## Start Postgres, MinIO and MLflow (persist across reboots)
	$(COMPOSE) up -d --build

services-down:  ## Stop the services (volumes are kept)
	$(COMPOSE) down

services-ps:  ## Show service status
	$(COMPOSE) ps

services-logs:  ## Tail service logs
	$(COMPOSE) logs -f --tail=50

check-services: image  ## Integration check -- requires `make services-up` first
	docker run --rm \
		--network rail-edge_default \
		--user $(HOST_UID):$(HOST_GID) -e HOME=/tmp \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		-v $(PWD):/workspace -w /workspace \
		$(IMAGE) python3 -m rail_edge_mlops.check_tracking

# --- Data versioning ------------------------------------------------------
# Joins the compose network because MinIO is bound to loopback on the host and
# is therefore unreachable from a bridge-network container. Credentials come
# from .env; .dvc/config holds only the bucket and endpoint, never secrets.
# Credentials come from .env locally, or from the environment in CI (where .env is
# absent because it is gitignored). The bare `-e VAR` form passes a variable through
# from the calling environment, so both paths work without a second code path.
DVC_RUN := docker run --rm \
	--network rail-edge_default \
	--user $(HOST_UID):$(HOST_GID) -e HOME=/tmp -e USER=$(HOST_USER) \
	$(if $(wildcard .env),--env-file .env,) \
	-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
	-v $(PWD):/workspace -w /workspace $(IMAGE)

.PHONY: dvc data-push data-pull

dvc:  ## Run a dvc command, e.g. make dvc CMD="status"
	$(DVC_RUN) dvc $(CMD)

data-push:  ## Upload tracked data to MinIO
	$(DVC_RUN) dvc push

data-pull:  ## Fetch tracked data from MinIO
	$(DVC_RUN) dvc pull

repro:  ## Rebuild the data pipeline (needs services up)
	$(DVC_RUN) dvc repro

check-repro:  ## Determinism gate: same inputs must produce the same outputs
	# Only the raw input. The whole point is to rebuild the derived outputs, so
	# fetching them first would be wasted bandwidth -- and would make the gate depend
	# on someone having remembered to push them.
	$(DVC_RUN) dvc pull data/raw/nuimages-v1.0-mini.tgz.dvc
	$(DVC_RUN) dvc repro --force
	@git diff --exit-code dvc.lock \
		&& echo "OK: pipeline is deterministic — dvc.lock unchanged" \
		|| { echo "FAIL: rerunning the pipeline changed dvc.lock (see diff above)"; exit 1; }

# --- Training -------------------------------------------------------------
# Needs both the GPU and MLflow, so it joins the compose network and carries the
# GPU flags. RAIL_EDGE_IMAGE is injected because a container cannot see the tag it
# was started from, and that tag is one third of a run's provenance.
TRAIN_RUN := docker run --rm $(GPU_ARGS) \
	--network rail-edge_default \
	-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
	-e RAIL_EDGE_IMAGE=$(IMAGE) \
	-e HF_HOME=/workspace/.cache/huggingface \
	$(if $(wildcard .env),--env-file .env,) \
	-v $(PWD):/workspace -w /workspace $(IMAGE)

.PHONY: train train-smoke

train: image  ## Fine-tune the detector (needs `make services-up`)
	$(TRAIN_RUN) python3 -m rail_edge_mlops.train $(ARGS)

train-smoke: image  ## 20 steps on a handful of images, to prove the loop runs
	$(TRAIN_RUN) python3 -m rail_edge_mlops.train \
		--experiment rtdetr-smoke --run-name smoke --allow-dirty $(ARGS)

models:  ## List registry versions with their provenance
	@docker run --rm --network rail-edge_default \
		--user $(HOST_UID):$(HOST_GID) -e HOME=/tmp -e USER=$(HOST_USER) \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		-v $(PWD):/workspace -w /workspace $(IMAGE) \
		python3 -c "from rail_edge_mlops import registry; import json; \
		print(json.dumps(registry.describe(), indent=2))"
