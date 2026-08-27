"""Build the train/val/test split -- grouped by capture log, held out by location.

Two properties matter, and both are easy to get wrong:

1. GROUPING. Images within one log are frames from the same drive, seconds apart and
   near-identical. Splitting at image level puts near-duplicates on both sides and inflates
   validation scores until the model meets real data. Every split boundary here falls
   between logs, never inside one.

2. HELD-OUT SHIFT. The test set is an entire location, not a random sample. Singapore and
   Boston differ in architecture, vegetation, vehicle mix -- and Singapore drives on the
   left. That makes the test set a genuine distribution shift rather than a reshuffle, which
   is what the drift work in P4 rests on.

Validation is carved from the *training* locations so the test location stays untouched by
hyperparameter choices. A test set you have tuned against is not held out.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def build_split(coco: dict, provenance: dict, test_location: str, val_fraction: float, seed: int):
    by_log: dict[str, list[int]] = defaultdict(list)
    log_location: dict[str, str] = {}
    for image_id_str, prov in provenance.items():
        by_log[prov["log_token"]].append(int(image_id_str))
        log_location[prov["log_token"]] = prov["location"]

    test_logs = sorted(t for t, loc in log_location.items() if loc == test_location)
    train_pool = sorted(t for t, loc in log_location.items() if loc != test_location)

    # Deterministic: same seed and same log set always yield the same partition.
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    n_val = max(1, round(len(train_pool) * val_fraction)) if train_pool else 0
    val_logs, train_logs = sorted(train_pool[:n_val]), sorted(train_pool[n_val:])

    return {"train": train_logs, "val": val_logs, "test": test_logs}, by_log, log_location


def subset(coco: dict, image_ids: set[int]) -> dict:
    images = [im for im in coco["images"] if im["id"] in image_ids]
    anns = [a for a in coco["annotations"] if a["image_id"] in image_ids]
    return {**coco, "images": images, "annotations": anns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    params = yaml.safe_load(args.params.read_text())["split"]
    coco = json.loads(args.coco.read_text())
    provenance = json.loads(args.provenance.read_text())
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    logs_by_split, by_log, log_location = build_split(
        coco, provenance, params["test_location"], params["val_fraction"], params["seed"]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy": (
            "Grouped by capture log; the test set is an entire held-out location. "
            "Validation is drawn from training locations so the test location is never "
            "tuned against."
        ),
        "params": params,
        "splits": {},
    }

    for split_name, logs in logs_by_split.items():
        image_ids = {i for t in logs for i in by_log[t]}
        sub = subset(coco, image_ids)
        (args.out_dir / f"{split_name}.json").write_text(json.dumps(sub))
        class_counts = Counter(id_to_name[a["category_id"]] for a in sub["annotations"])
        manifest["splits"][split_name] = {
            "logs": logs,
            "locations": sorted({log_location[t] for t in logs}),
            "n_logs": len(logs),
            "n_images": len(sub["images"]),
            "n_boxes": len(sub["annotations"]),
            "class_counts": dict(sorted(class_counts.items())),
        }
        print(
            f"{split_name:<6} {len(logs):>3} logs  {len(sub['images']):>5} images  "
            f"{len(sub['annotations']):>6} boxes  {sorted({log_location[t] for t in logs})}"
        )

    # The invariant this whole module exists to guarantee.
    all_logs = [t for logs in logs_by_split.values() for t in logs]
    assert len(all_logs) == len(set(all_logs)), "a log appears in more than one split"

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
