"""nuImages -> COCO detection format.

COCO because every detection framework reads it, which keeps the model choice swappable.
Alongside the annotations we emit a sidecar of per-image provenance -- log token, location,
capture time -- because the split policy needs it and it is otherwise buried in nuImages'
relational metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from rail_edge_mlops.data.categories import CLASS_TO_ID, DETECTION_CLASSES, NUIMAGES_TO_DETECTION

# Capture time is not a field; it is encoded in the logfile name, tz-aware:
#   n003-2018-01-03-12-03-23+0800
LOGFILE_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})([+-]\d{4})")


def _load(root: Path, name: str) -> list[dict]:
    return json.loads((root / f"{name}.json").read_text())


def _local_hour(logfile: str) -> int | None:
    m = LOGFILE_TS.match(logfile.split("-", 1)[1] if logfile[0].isalpha() else logfile)
    if not m:
        m = LOGFILE_TS.search(logfile)
    return int(m.group(4)) if m else None


def convert(meta_dir: Path, out_path: Path, sidecar_path: Path) -> dict:
    logs = {log["token"]: log for log in _load(meta_dir, "log")}
    samples = _load(meta_dir, "sample")
    sample_data = {sd["token"]: sd for sd in _load(meta_dir, "sample_data")}
    categories = {c["token"]: c["name"] for c in _load(meta_dir, "category")}
    object_anns = _load(meta_dir, "object_ann")

    images: list[dict] = []
    provenance: dict[str, dict] = {}
    sd_to_image_id: dict[str, int] = {}

    for sample in samples:
        sd = sample_data[sample["key_camera_token"]]
        log = logs[sample["log_token"]]
        image_id = len(images) + 1
        sd_to_image_id[sd["token"]] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": sd["filename"],
                "width": sd["width"],
                "height": sd["height"],
            }
        )
        provenance[str(image_id)] = {
            "log_token": sample["log_token"],
            "location": log["location"],
            "logfile": log["logfile"],
            "local_hour": _local_hour(log["logfile"]),
            "date_captured": log["date_captured"],
            "timestamp": sample["timestamp"],
        }

    annotations: list[dict] = []
    kept, dropped = Counter(), Counter()

    for ann in object_anns:
        image_id = sd_to_image_id.get(ann["sample_data_token"])
        if image_id is None:
            continue  # annotation on a non-key frame
        source = categories[ann["category_token"]]
        target = NUIMAGES_TO_DETECTION.get(source)
        if target is None:
            dropped[source] += 1
            continue
        x1, y1, x2, y2 = ann["bbox"]
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": CLASS_TO_ID[target],
                # COCO wants xywh, nuImages gives xyxy
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
            }
        )
        kept[target] += 1

    coco = {
        "info": {
            "description": "nuImages converted to COCO, nuScenes 10-class detection taxonomy",
            "source": "nuImages by Motional, CC BY-NC-SA 4.0",
            "generated": datetime.now().isoformat(timespec="seconds"),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CLASS_TO_ID[n], "name": n} for n in DETECTION_CLASSES],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco))
    sidecar_path.write_text(json.dumps(provenance, indent=2))

    return {"images": len(images), "kept": dict(kept), "dropped": dict(dropped)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    args = ap.parse_args()

    stats = convert(args.meta_dir, args.out, args.sidecar)
    total_kept = sum(stats["kept"].values())
    total_dropped = sum(stats["dropped"].values())
    print(f"images     : {stats['images']}")
    print(f"kept       : {total_kept} boxes across {len(stats['kept'])} classes")
    for name, n in sorted(stats["kept"].items(), key=lambda kv: -kv[1]):
        print(f"   {name:<22} {n}")
    print(f"dropped    : {total_dropped} boxes in unmapped categories")
    for name, n in sorted(stats["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"   {name:<38} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
