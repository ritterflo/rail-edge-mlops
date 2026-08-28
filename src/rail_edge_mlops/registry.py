"""Register trained models as versions, so promotion is something that can be gated.

Logging an artifact to a run records *what happened*. A registry records *what is
current* -- a named model with ordered versions and a stage each version sits in. P2's
gate blocks a stage transition, which requires the stage to exist.

Each version carries the provenance triple from its run, so "which code, data and
environment produced the model currently in production?" is answerable from the registry
alone, without knowing which run to look at.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "rail-edge-detector"


def register(run_id: str, name: str = MODEL_NAME, artifact_path: str = "model") -> str:
    """Create a new version of `name` from a run's logged artifact.

    Returns the version number. The provenance tags are copied from the run onto the
    version: a registry entry that cannot say what produced it is not much better than a
    file on disk.
    """
    client = MlflowClient()
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    source = f"runs:/{run_id}/{artifact_path}"
    version = client.create_model_version(name=name, source=source, run_id=run_id)

    run = client.get_run(run_id)
    for key in ("git_commit", "git_branch", "git_dirty", "data_lock_md5", "image_tag"):
        if key in run.data.tags:
            client.set_model_version_tag(name, version.version, key, run.data.tags[key])

    # The operating threshold does not transfer between versions, so it belongs on the
    # version rather than in a serving config. See reports/threshold_transfer.json.
    for metric in ("val.map", "val.map_small"):
        if metric in run.data.metrics:
            client.set_model_version_tag(
                name, version.version, metric, str(run.data.metrics[metric])
            )

    return version.version


def describe(name: str = MODEL_NAME) -> list[dict]:
    """Every version, with the provenance needed to reproduce it."""
    client = MlflowClient()
    out = []
    for v in client.search_model_versions(f"name='{name}'"):
        out.append(
            {
                "version": v.version,
                "run_id": v.run_id,
                "aliases": list(v.aliases or []),
                "git_commit": (v.tags or {}).get("git_commit", "")[:12],
                "data_lock_md5": (v.tags or {}).get("data_lock_md5", "")[:12],
                "image_tag": (v.tags or {}).get("image_tag", ""),
                "val_map": (v.tags or {}).get("val.map", ""),
            }
        )
    return sorted(out, key=lambda d: int(d["version"]))
