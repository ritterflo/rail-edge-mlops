"""Tests for the run-provenance record.

The dirty-worktree refusal is the one with teeth: without it a run records a commit
that does not describe the code that ran, and the whole lineage claim is decorative.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rail_edge_mlops import provenance


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    (tmp_path / "a.txt").write_text("one\n")
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    return tmp_path


def test_clean_worktree_yields_the_triple(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("RAIL_EDGE_IMAGE", "rail-edge/train:abc123")

    prov = provenance.collect()
    assert len(prov["git_commit"]) == 40
    assert prov["git_dirty"] == "false"
    assert len(prov["data_lock_md5"]) == 32
    assert prov["image_tag"] == "rail-edge/train:abc123"


def test_dirty_worktree_refuses(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "a.txt").write_text("changed\n")

    with pytest.raises(provenance.DirtyWorktreeError, match="dirty worktree"):
        provenance.collect()


def test_dirty_worktree_can_be_overridden(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "a.txt").write_text("changed\n")

    prov = provenance.collect(allow_dirty=True)
    assert prov["git_dirty"] == "true", "an overridden run must still admit it was dirty"


def test_data_hash_tracks_the_lock_file(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    before = provenance.collect()["data_lock_md5"]

    (repo / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n")
    subprocess.run(["git", "-C", str(repo), "commit", "-aqm", "data"], check=True)
    assert provenance.collect()["data_lock_md5"] != before


def test_missing_image_tag_is_explicit(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("RAIL_EDGE_IMAGE", raising=False)
    assert provenance.collect()["image_tag"] == "unknown"


def test_describe_flags_dirty_runs(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "a.txt").write_text("changed\n")
    assert "DIRTY" in provenance.describe(provenance.collect(allow_dirty=True))
