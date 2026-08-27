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
DOCKER_RUN := docker run --rm \
	--device=/dev/kfd --device=/dev/dri \
	--group-add video --group-add render \
	--security-opt seccomp=unconfined \
	--ipc=host --shm-size=8g \
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
	@docker run --rm -it \
		--device=/dev/kfd --device=/dev/dri \
		--group-add video --group-add render \
		--security-opt seccomp=unconfined \
		--ipc=host --shm-size=8g \
		-v $(PWD):/workspace -w /workspace $(IMAGE) /bin/bash

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
