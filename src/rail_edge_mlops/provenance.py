"""What produced this model.

Code, environment, data and config are each independently reproducible in this repo,
but nothing yet recorded *which combination* produced a given artefact. Without that,
four versioned things still cannot answer "why did this model regress?".

Three identifiers together pin a run completely:

    git commit    the code and the config in params.yaml
    dvc.lock md5  the exact data every stage produced
    image tag     the environment, itself a hash of the Dockerfile and uv.lock

A dirty worktree breaks the first of those -- the commit would describe code that is
not what ran -- so training refuses to start on one unless explicitly overridden.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


class DirtyWorktreeError(RuntimeError):
    pass


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _file_md5(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect(allow_dirty: bool = False) -> dict[str, str]:
    """Gather the identifiers that pin this run. Raises on a dirty worktree."""
    dirty_files = _git("status", "--porcelain")

    if dirty_files and not allow_dirty:
        listed = "\n  ".join(dirty_files.splitlines()[:10])
        raise DirtyWorktreeError(
            "Refusing to train on a dirty worktree: the recorded commit would not "
            "describe the code that actually ran, which makes the run unreproducible.\n"
            f"  {listed}\n"
            "Commit the changes, or pass --allow-dirty for a throwaway run."
        )

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(dirty_files)).lower(),
        # Not the file's own hash but a hash of the hashes it records -- so it changes
        # when the data changes and not when a comment in dvc.yaml does.
        "data_lock_md5": _file_md5(Path("dvc.lock")),
        # Injected by the Makefile; a container cannot see the tag it was started from.
        "image_tag": os.environ.get("RAIL_EDGE_IMAGE", "unknown"),
    }


def describe(prov: dict[str, str]) -> str:
    return (
        f"  code        : {prov['git_commit'][:12]} ({prov['git_branch']})"
        f"{'  DIRTY' if prov['git_dirty'] == 'true' else ''}\n"
        f"  data        : dvc.lock {prov['data_lock_md5'][:12]}\n"
        f"  environment : {prov['image_tag']}"
    )
