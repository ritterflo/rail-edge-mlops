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


def _greedy_match(pred: dict, gt: dict, iou_threshold: float, limit: int) -> list[tuple]:
    """Match the `limit` highest-scoring detections to ground truth, best score first.

    Mirrors how a consumer reads detections: the confident one claims the object, and a
    ground-truth box can only be claimed once.
    """
    out = []
    claimed = torch.zeros(len(gt["boxes"]), dtype=torch.bool)
    order = torch.argsort(pred["scores"], descending=True)[:limit]
    for i in order:
        box, label, score = pred["boxes"][i], pred["labels"][i], pred["scores"][i]
        eligible = (gt["labels"] == label) & ~claimed
        hit = 0
        if eligible.any():
            ious = _iou(box, gt["boxes"][eligible])
            best = int(torch.argmax(ious))
            if ious[best] >= iou_threshold:
                claimed[torch.nonzero(eligible).flatten()[best]] = True
                hit = 1
        out.append((float(score), hit))
    return out


def operating_points(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
) -> list[dict]:
    """Precision, recall and detection density at candidate score thresholds.

    This is what actually answers "where do I set the threshold" for serving. mAP cannot:
    it is threshold-free by construction. A single calibration error cannot either: it
    compresses exactly the trade-off you need to see into one number.
    """
    n_gt = sum(len(gt["boxes"]) for gt in targets)
    n_images = max(len(targets), 1)
    rows = []
    for t in thresholds:
        tp = n_det = 0
        for pred, gt in zip(predictions, targets, strict=True):
            keep = pred["scores"] >= t
            kept = {k: v[keep] for k, v in pred.items()}
            matches = _greedy_match(kept, gt, iou_threshold, limit=len(kept["scores"]))
            tp += sum(h for _, h in matches)
            n_det += len(matches)
        rows.append(
            {
                "threshold": t,
                "detections_per_image": round(n_det / n_images, 3),
                "precision": round(tp / n_det, 4) if n_det else 0.0,
                "recall": round(tp / n_gt, 4) if n_gt else 0.0,
            }
        )
    return rows


def calibration(
    predictions: list[dict], targets: list[dict], iou_threshold: float = 0.5, bins: int = 10
) -> dict:
    """Confidence against observed precision, over top-K per image where K is that
    image's ground-truth count.

    K comes from the data, not from the model's scores. Two earlier attempts -- a fixed
    score threshold, then a fixed top-20 -- both let the measured population depend on
    where the score distribution happened to sit, which is why ECE varied by 0.135
    across three identical runs while mAP varied by 0.0066.

    The bin table is the useful output: it is what recalibration consumes, and
    "confidence 0.13 means precision 0.84" is directly actionable. The ECE below it is
    retained only to compare before and after a recalibration fitted on the *same*
    predictions -- it is not comparable across runs.
    """
    scored: list[tuple[float, int]] = []
    all_scores: list[float] = []

    for pred, gt in zip(predictions, targets, strict=True):
        all_scores.extend(pred["scores"].tolist())
        scored.extend(_greedy_match(pred, gt, iou_threshold, limit=len(gt["boxes"])))

    st = torch.tensor(all_scores)
    percentiles = (
        {f"p{q}": round(float(torch.quantile(st, q / 100)), 4) for q in (50, 90, 99)}
        if st.numel()
        else {}
    )
    percentiles["max"] = round(float(st.max()), 4) if st.numel() else 0.0

    if not scored:
        return {
            "ece_same_predictions_only": 0.0,
            "n_measured": 0,
            "bins": [],
            "score_percentiles": percentiles,
        }

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

    return {
        "ece_same_predictions_only": round(ece, 4),
        "n_measured": total,
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
        "operating_points": operating_points(all_preds, all_targets),
    }


def summarise(name: str, result: dict) -> str:
    cal = result["calibration"]
    pct = cal.get("score_percentiles", {})
    lines = [
        f"{name}:",
        f"  mAP        {result['map']:.4f}"
        f"   @50 {result['map_50']:.4f}   @75 {result['map_75']:.4f}",
        f"  by size    small {result['map_small']:.4f}  medium {result['map_medium']:.4f}"
        f"  large {result['map_large']:.4f}",
        f"  scores     p50 {pct.get('p50', 0):.3f}  p90 {pct.get('p90', 0):.3f}"
        f"  p99 {pct.get('p99', 0):.3f}  max {pct.get('max', 0):.3f}",
    ]
    for row in result.get("operating_points", []):
        lines.append(
            f"    thr {row['threshold']:.2f}   {row['detections_per_image']:>6.2f} det/img"
            f"   precision {row['precision']:.3f}   recall {row['recall']:.3f}"
        )
    if result["ap_per_class"]:
        worst = sorted(result["ap_per_class"].items(), key=lambda kv: kv[1])[:3]
        lines.append("  weakest    " + ", ".join(f"{k} {v:.3f}" for k, v in worst))
    return "\n".join(lines)


def run_metrics(split: str, result: dict) -> dict[str, float]:
    """The scalar view of an `evaluate()` result, named `<split>.<key>` for MLflow.

    The ECE is logged under its full name rather than `ece`: the name carries the caveat
    that it only compares before and after a recalibration on the same predictions.
    """
    flat = {f"{split}.{k}": v for k, v in result.items() if isinstance(v, (int, float))}
    flat[f"{split}.ece_same_predictions_only"] = result["calibration"]["ece_same_predictions_only"]
    return flat


def write(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
