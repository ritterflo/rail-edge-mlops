"""Integration check: prove MLflow, Postgres and MinIO are wired to each other.

Unlike the unit tests, this deliberately requires the real services (`make
services-up`). Three containers all showing "healthy" proves only that they
started -- not that a client can log a run, that metadata reaches Postgres, or
that artifact bytes reach MinIO. So this logs a run and reads it back.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow


def main() -> int:
    print(f"tracking uri : {os.environ.get('MLFLOW_TRACKING_URI')}")

    mlflow.set_experiment("service-check")
    with mlflow.start_run(run_name="wiring-check") as run:
        run_id = run.info.run_id
        mlflow.log_param("purpose", "verify postgres + minio wiring")
        mlflow.log_metric("answer", 42.0)
        with tempfile.TemporaryDirectory() as tmp:
            proof = Path(tmp) / "proof.txt"
            proof.write_text("if you can read this back, the wiring works\n")
            mlflow.log_artifact(str(proof))

    print(f"run id       : {run_id}")

    client = mlflow.MlflowClient()

    # Metadata round trip -> Postgres
    stored = client.get_run(run_id)
    assert stored.data.params.get("purpose"), "param never reached the backend store"
    assert stored.data.metrics.get("answer") == 42.0, "metric never reached the backend store"

    # Artifact round trip -> MinIO
    names = [a.path for a in client.list_artifacts(run_id)]
    print(f"artifacts    : {names}")
    assert "proof.txt" in names, "artifact never reached the artifact store"

    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="proof.txt", dst_path=tmp
        )
        print(f"round trip   : {Path(local).read_text().strip()!r}")

    print("OK: metadata in Postgres, artifact bytes in MinIO, both readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
