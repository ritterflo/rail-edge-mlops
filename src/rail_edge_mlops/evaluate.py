"""Evaluate a detector: mAP, per-class AP, and confidence calibration.

mAP says how good the detections are. Calibration says whether the confidence attached
to them means anything -- and those are different questions. RT-DETR has no objectness
head (focal loss, no background class), so the max class score is the only signal
available for "is this real?", and out of the box it is a ranking, not a probability.

Expected Calibration Error bins detections by score and compares each bin's mean score
against the fraction of that bin which actually matched a ground-truth box. ECE 0.30
means confidences are off by 30 percentage points on average -- so a 0.9 detection is
not a 90% one, and any threshold chosen from those numbers is chosen from fiction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision


def _iou(box: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """IoU of one xyxy box against many."""
    if others.numel() == 0:
        return torch.zeros(0)
    x1 = torch.maximum(box[0], others[:, 0])
    y1 = torch.maximum(box[1], others[:, 1])
    x2 = torch.minimum(box[2], others[:, 2])
    y2 = torch.minimum(box[3], others[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_o = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_b + area_o - inter + 1e-9)


def calibration(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    bins: int = 10,
    top_k: int = 20,
) -> dict:
    """Bin the top-K detections per image by score; compare score against hit rate.

    Top-K per image rather than everything above a score threshold, deliberately. With a
    fixed threshold the population being measured depends on where the score
    distribution happens to sit -- three identical training runs produced 66,707,
    68,452 and 113,386 detections above 0.05, and since ECE is a population-weighted
    average over bins, its standard deviation across those runs was 0.135 against
    mAP's 0.0066. That was measuring threshold placement, not calibration.

    K=20 sits well above the 6.75 boxes/image in the ground truth and well below COCO's
    conventional 100, which on this data would be mostly near-zero padding.
    """
    scored: list[tuple[float, int]] = []
    all_scores: list[float] = []

    for pred, gt in zip(predictions, targets, strict=True):
        all_scores.extend(pred["scores"].tolist())
        gt_boxes, gt_labels = gt["boxes"], gt["labels"]
        claimed = torch.zeros(len(gt_boxes), dtype=torch.bool)
        # Greedy highest-score-first matching, mirroring how a consumer would read
        # these detections: the confident one gets the object.
        for i in torch.argsort(pred["scores"], descending=True)[:top_k]:
            box, label, score = pred["boxes"][i], pred["labels"][i], pred["scores"][i]
            eligible = (gt_labels == label) & ~claimed
            hit = 0
            if eligible.any():
                ious = _iou(box, gt_boxes[eligible])
                best = int(torch.argmax(ious))
                if ious[best] >= iou_threshold:
                    claimed[torch.nonzero(eligible).flatten()[best]] = True
                    hit = 1
            scored.append((float(score), hit))

    if not scored:
        return {"ece": 0.0, "n_detections": 0, "bins": [], "score_percentiles": {}}

    edges = torch.linspace(0, 1, bins + 1)
    buckets = defaultdict(list)
    for score, hit in scored:
        idx = min(int(torch.bucketize(torch.tensor(score), edges)) - 1, bins - 1)
        buckets[max(idx, 0)].append((score, hit))

    total = len(scored)
    ece = 0.0
    rows = []
    for b in range(bins):
        items = buckets.get(b, [])
        if not items:
            continue
        mean_score = sum(s for s, _ in items) / len(items)
        precision = sum(h for _, h in items) / len(items)
        ece += len(items) / total * abs(mean_score - precision)
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": len(items),
                "mean_confidence": round(mean_score, 4),
                "observed_precision": round(precision, 4),
                "gap": round(mean_score - precision, 4),
            }
        )

    # The compression is currently something you infer from bin populations. Reported
    # directly, it is a number: if p99 sits near 0.2, no conventional threshold works.
    st = torch.tensor(all_scores)
    percentiles = (
        {f"p{q}": round(float(torch.quantile(st, q / 100)), 4) for q in (50, 90, 99)}
        if st.numel()
        else {}
    )
    percentiles["max"] = round(float(st.max()), 4) if st.numel() else 0.0

    return {
        "ece": round(ece, 4),
        "n_detections": total,
        "top_k_per_image": top_k,
        "n_detections_above_threshold": len(all_scores),
        "score_percentiles": percentiles,
        "bins": rows,
    }


@torch.no_grad()
def evaluate(model, processor, loader, device, score_threshold: float = 0.05) -> dict:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    all_preds: list[dict] = []
    all_targets: list[dict] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)

        sizes = torch.tensor(
            [[lbl["size"][0], lbl["size"][1]] for lbl in batch["labels"]], device=device
        )
        # Post-process once with no score cut, then take two different views of it.
        # Calibration needs an unfiltered population so top-K is the only thing
        # selecting detections; mAP keeps the score threshold so it stays comparable
        # with the recorded noise floor. Filtering first and capping second -- which is
        # what a threshold plus top-K does -- leaves the population set by the
        # threshold after all.
        results = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=sizes
        )

        targets = []
        for lbl, size in zip(batch["labels"], sizes, strict=True):
            h, w = float(size[0]), float(size[1])
            # Labels come back as normalised cxcywh; mAP and IoU want absolute xyxy.
            boxes = lbl["boxes"].clone()
            cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
            targets.append(
                {
                    "boxes": torch.stack(
                        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1
                    ),
                    "labels": lbl["class_labels"],
                }
            )

        cpu_preds = [{k: v.cpu() for k, v in r.items()} for r in results]
        thresholded = [
            {k: v[p["scores"] >= score_threshold] for k, v in p.items()} for p in cpu_preds
        ]
        metric.update(thresholded, targets)
        all_preds.extend(cpu_preds)
        all_targets.extend(targets)

    computed = metric.compute()
    id_to_name = {c["id"]: c["name"] for c in loader.dataset.categories}

    per_class = {}
    if "classes" in computed and computed["classes"].numel():
        for cls, ap in zip(
            computed["classes"].tolist(), computed["map_per_class"].tolist(), strict=False
        ):
            per_class[id_to_name.get(int(cls), str(cls))] = round(float(ap), 4)

    return {
        "map": round(float(computed["map"]), 4),
        "map_50": round(float(computed["map_50"]), 4),
        "map_75": round(float(computed["map_75"]), 4),
        "map_small": round(float(computed["map_small"]), 4),
        "map_medium": round(float(computed["map_medium"]), 4),
        "map_large": round(float(computed["map_large"]), 4),
        "ap_per_class": per_class,
        "calibration": calibration(all_preds, all_targets),
    }


def summarise(name: str, result: dict) -> str:
    cal = result["calibration"]
    lines = [
        f"{name}:",
        f"  mAP        {result['map']:.4f}"
        f"   @50 {result['map_50']:.4f}   @75 {result['map_75']:.4f}",
        f"  by size    small {result['map_small']:.4f}  medium {result['map_medium']:.4f}"
        f"  large {result['map_large']:.4f}",
        f"  ECE        {cal['ece']:.4f} over {cal['n_detections']:,} detections",
    ]
    if result["ap_per_class"]:
        worst = sorted(result["ap_per_class"].items(), key=lambda kv: kv[1])[:3]
        lines.append("  weakest    " + ", ".join(f"{k} {v:.3f}" for k, v in worst))
    return "\n".join(lines)


def write(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
