"""Fine-tune RT-DETRv2 on the nuImages splits, recording what produced the result.

The point of this script is not the training loop -- it is that every run states, in a
form someone else can check, which code, which data and which environment produced it.
A model whose provenance is unknown is not reproducible no matter how carefully it was
trained.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.multiprocessing
import yaml
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from rail_edge_mlops import provenance
from rail_edge_mlops.data.categories import CLASS_TO_ID, DETECTION_CLASSES
from rail_edge_mlops.data.dataset import CocoDetection, collate
from rail_edge_mlops.evaluate import evaluate, summarise, write

# DataLoader workers pass tensors between processes as file descriptors by default,
# and the container's soft limit is 1024. Eight workers over thousands of batches
# exhausts that and fails with "received 0 items of ancdata" -- typically far into a
# run. The file_system strategy passes shared-memory names instead and has no such cap.
torch.multiprocessing.set_sharing_strategy("file_system")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimiser(model, cfg: dict) -> torch.optim.Optimizer:
    """Lower learning rate on the backbone than on the heads.

    The classification head starts from random weights and must move a long way; the
    backbone already encodes what objects look like and mostly needs to stay put. One
    learning rate for both either moves the head too slowly or wrecks the backbone.
    """
    backbone, rest = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone if "backbone" in name else rest).append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": cfg["lr"] * cfg["backbone_lr_mult"]},
            {"params": rest, "lr": cfg["lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )


def lr_lambda(step: int, warmup: int, total: int):
    """Linear warmup, then cosine decay.

    Warmup exists because the randomly-initialised head produces large, meaningless
    gradients for the first few hundred steps. Applying them at full learning rate is
    the classic way to destroy pretrained features before they are ever used.
    """
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    ap.add_argument("--images-root", type=Path, default=Path("data/interim/nuimages"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--experiment", default="rtdetr-baseline")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    # Overrides for smoke runs and bisection. Anything passed here is logged to
    # MLflow with the rest of the config, so an overridden run cannot be mistaken
    # for one that used params.yaml as committed.
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument(
        "--limit-images", type=int, default=None, help="use only the first N training images"
    )
    ap.add_argument(
        "--limit-eval-images", type=int, default=None, help="evaluate on only the first N images"
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training; load the model from --out-dir and evaluate it. Evaluates an "
        "existing checkpoint against a split without spending another epoch, and "
        "recovers a run interrupted after the model was saved.",
    )
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="also evaluate the held-out location; off by default so the "
        "shifted split is not tuned against by habit",
    )
    args = ap.parse_args()

    # Fail before anything expensive if the run could not be described afterwards.
    prov = provenance.collect(allow_dirty=args.allow_dirty)
    print("provenance:")
    print(provenance.describe(prov))

    cfg = yaml.safe_load(args.params.read_text())["train"]
    for key, value in (
        ("max_steps", args.max_steps),
        ("batch_size", args.batch_size),
        ("num_workers", args.num_workers),
    ):
        if value is not None:
            cfg[key] = value
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible; this will be extremely slow.")

    processor = AutoImageProcessor.from_pretrained(
        cfg["checkpoint"],
        revision=cfg["checkpoint_revision"],
        size={"height": cfg["image_size"][0], "width": cfg["image_size"][1]},
    )
    source = str(args.out_dir / "model") if args.eval_only else cfg["checkpoint"]
    model = AutoModelForObjectDetection.from_pretrained(
        source,
        revision=cfg["checkpoint_revision"],
        num_labels=len(DETECTION_CLASSES),
        id2label={v: k for k, v in CLASS_TO_ID.items()},
        label2id=CLASS_TO_ID,
        # The COCO head is 80-wide and ours is 10; shapes cannot match, so it is
        # discarded and reinitialised. Everything else -- backbone, encoder, decoder,
        # and the class-independent box head -- loads unchanged.
        ignore_mismatched_sizes=True,
    ).to(device)

    loaders = {}
    for split in ("train", "val", "test"):
        path = args.splits_dir / f"{split}.json"
        if not path.exists():
            continue
        ds = CocoDetection(path, args.images_root, processor)
        if args.limit_images and split == "train":
            ds.images = ds.images[: args.limit_images]
        if args.limit_eval_images and split != "train":
            ds.images = ds.images[: args.limit_eval_images]
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["num_workers"],
            collate_fn=collate,
            drop_last=(split == "train"),
            persistent_workers=cfg["num_workers"] > 0,
        )
        print(f"{split:<6} {len(ds):>7,} images")

    steps_per_epoch = len(loaders["train"])
    total_steps = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    optimiser = build_optimiser(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: lr_lambda(s, cfg["warmup_steps"], total_steps)
    )

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        # The triple, logged as tags so runs can be filtered on them in the UI.
        mlflow.set_tags(prov)
        mlflow.log_params({f"train.{k}": v for k, v in cfg.items()})
        mlflow.log_param("n_train_images", len(loaders["train"].dataset))

        model.train()
        step = 0
        running = 0.0
        if args.eval_only:
            print("eval-only: skipping training")
            total_steps = 0
        for _epoch in range(0 if args.eval_only else cfg["epochs"]):
            for batch in loaders["train"]:
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]

                loss = model(pixel_values=pixel_values, labels=labels).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)

                running += loss.item()
                step += 1
                if step % 20 == 0:
                    mean = running / 20
                    mlflow.log_metric("train_loss", mean, step=step)
                    mlflow.log_metric("lr", scheduler.get_last_lr()[-1], step=step)
                    print(f"  step {step:>6}/{total_steps}  loss {mean:.4f}")
                    running = 0.0
                if step >= total_steps:
                    break
            if step >= total_steps:
                break

        if not args.eval_only:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out_dir / "model")
            processor.save_pretrained(args.out_dir / "model")

        splits_to_eval = ["val"] + (["test"] if args.eval_test else [])
        for split in splits_to_eval:
            if split not in loaders:
                continue
            print(f"\nevaluating {split}...")
            result = evaluate(model, processor, loaders[split], device, cfg["score_threshold"])
            print(summarise(split, result))
            write(result, args.out_dir / f"eval_{split}.json")
            mlflow.log_metrics(
                {f"{split}.{k}": v for k, v in result.items() if isinstance(v, (int, float))}
            )
            mlflow.log_metric(f"{split}.ece", result["calibration"]["ece"])
            mlflow.log_dict(result, f"eval_{split}.json")

        mlflow.log_artifacts(str(args.out_dir / "model"), artifact_path="model")
        print(f"\nrun: {mlflow.active_run().info.run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
