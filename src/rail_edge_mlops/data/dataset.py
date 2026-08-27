"""COCO-format dataset feeding RT-DETR's image processor.

The processor owns resizing, normalisation and the box-format conversion the model
expects, so this class stays thin: read the split JSON, open the image, hand both over.
Keeping it thin matters -- a hand-rolled preprocessing path that disagrees with the
checkpoint's own is a silent accuracy loss that looks like a training problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoDetection(Dataset):
    def __init__(self, coco_path: Path, images_root: Path, processor) -> None:
        coco = json.loads(Path(coco_path).read_text())
        self.images = coco["images"]
        self.images_root = Path(images_root)
        self.processor = processor
        self.categories = coco["categories"]
        self.num_classes = len(self.categories)

        by_image: dict[int, list[dict]] = {im["id"]: [] for im in self.images}
        for ann in coco["annotations"]:
            by_image[ann["image_id"]].append(ann)
        self.annotations = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        record = self.images[idx]
        image = Image.open(self.images_root / record["file_name"]).convert("RGB")
        target = {"image_id": record["id"], "annotations": self.annotations[record["id"]]}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        labels = encoding["labels"][0]

        # Fail here, in Python, rather than inside a GPU kernel. An out-of-range class
        # label indexes past the classification logits and aborts the queue with an
        # opaque HSA exception -- and only when a batch happens to contain the offending
        # class, which makes it look like a batch-size or hardware problem.
        if labels["class_labels"].numel():
            worst = int(labels["class_labels"].max())
            if worst >= self.num_classes:
                raise ValueError(
                    f"class label {worst} is out of range for a {self.num_classes}-class "
                    f"model (valid 0..{self.num_classes - 1}); image "
                    f"{record['file_name']}. Category ids must be zero-indexed."
                )

        return {
            "pixel_values": encoding["pixel_values"][0],
            "labels": labels,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }
