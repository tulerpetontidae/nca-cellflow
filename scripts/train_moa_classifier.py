"""
Train an Inception-based MoA (Mode of Action) classifier on real treated images.

The trained classifier is used to evaluate whether generated images preserve
the correct MoA morphology. Follows the CellFlux evaluation protocol:
  - Frozen Inception-v3 features (2048-dim, same backbone as FID)
  - 2-layer MLP head: 2048 → 512 → num_moa_classes
  - Trained on real treated images from the train split
  - Evaluated on real treated images from the test split

Supports OOD split: when --exclude_compounds is set, those compounds are
removed from training, and a separate OOD evaluation is reported.

Usage:
    python scripts/train_moa_classifier.py
    python scripts/train_moa_classifier.py --exclude_compounds docetaxel AZ841 ...
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score

from nca_cellflow.dataset import _load_image
from nca_cellflow.models import MOAClassifier


class MOADataset(Dataset):
    """Treated images labeled by MoA (ANNOT column)."""

    def __init__(self, metadata_csv: str, image_dir: str, split: str = "train",
                 image_size: int = 96, augment: bool = True,
                 exclude_compounds: list[str] | None = None,
                 only_compounds: list[str] | None = None,
                 moa2id: dict[str, int] | None = None):
        df = pd.read_csv(metadata_csv, index_col=0)
        df = df[(df["SPLIT"] == split) & (df["STATE"] == 1)]

        if exclude_compounds:
            df = df[~df["CPD_NAME"].isin(exclude_compounds)]
        if only_compounds:
            df = df[df["CPD_NAME"].isin(only_compounds)]

        # Build MoA label space (from provided mapping or from this split)
        if moa2id is not None:
            self.moa2id = moa2id
        else:
            moas = sorted(df["ANNOT"].unique())
            self.moa2id = {m: i for i, m in enumerate(moas)}
        self.id2moa = {i: m for m, i in self.moa2id.items()}
        self.num_classes = len(self.moa2id)

        # Filter to only compounds whose MoA is in the label space
        df = df[df["ANNOT"].isin(self.moa2id)]

        self.keys = df["SAMPLE_KEY"].values
        self.moa_labels = np.array([self.moa2id[a] for a in df["ANNOT"].values])
        self.cpd_names = df["CPD_NAME"].values

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img = _load_image(self.image_dir, self.keys[idx],
                          self.image_size, augment=self.augment)
        # Convert from [-1, 1] to [0, 1] for Inception
        img = (img + 1) / 2
        label = self.moa_labels[idx]
        return img, label


def evaluate(model, loader, device):
    """Compute accuracy, macro-F1, weighted-F1, and per-class accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    p = argparse.ArgumentParser(description="Train MoA classifier on real treated images")
    p.add_argument("--metadata_csv", type=str,
                   default="data/bbbc021_six/metadata/bbbc021_df_all.csv")
    p.add_argument("--image_dir", type=str, default="data/bbbc021_six")
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--exclude_compounds", type=str, nargs="+", default=None,
                   help="Compounds to exclude from training (OOD split)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/moa_classifier")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Datasets
    train_ds = MOADataset(
        args.metadata_csv, args.image_dir, split="train",
        image_size=args.image_size, augment=True,
        exclude_compounds=args.exclude_compounds,
    )
    test_ds = MOADataset(
        args.metadata_csv, args.image_dir, split="test",
        image_size=args.image_size, augment=False,
        exclude_compounds=args.exclude_compounds,
        moa2id=train_ds.moa2id,
    )
    print(f"Train: {len(train_ds)} images, {train_ds.num_classes} MoA classes")
    print(f"Test:  {len(test_ds)} images")
    print(f"MoA classes: {list(train_ds.moa2id.keys())}")

    if args.exclude_compounds:
        print(f"Excluded compounds: {args.exclude_compounds}")
        # Also create OOD test set (only excluded compounds)
        ood_ds = MOADataset(
            args.metadata_csv, args.image_dir, split="test",
            image_size=args.image_size, augment=False,
            only_compounds=args.exclude_compounds,
            moa2id=train_ds.moa2id,
        )
        print(f"OOD test: {len(ood_ds)} images")
    else:
        ood_ds = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    if ood_ds is not None and len(ood_ds) > 0:
        ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        ood_loader = None

    # Model
    model = MOAClassifier(num_classes=train_ds.num_classes).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Count params
    n_trainable = sum(p.numel() for p in model.classifier.parameters())
    print(f"Trainable parameters: {n_trainable:,}")

    # Training
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        # Keep Inception frozen
        model._fid.inception.eval()

        total_loss = 0.0
        n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        line = (f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
                f"test_acc={test_metrics['accuracy']:.4f}  "
                f"macro_f1={test_metrics['macro_f1']:.4f}  "
                f"weighted_f1={test_metrics['weighted_f1']:.4f}")

        if ood_loader is not None:
            ood_metrics = evaluate(model, ood_loader, device)
            line += (f"  ood_acc={ood_metrics['accuracy']:.4f}  "
                     f"ood_f1={ood_metrics['macro_f1']:.4f}")

        print(line)

        # Save best
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            save_path = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch + 1,
                "classifier_state": model.state_dict_head(),
                "moa2id": train_ds.moa2id,
                "id2moa": train_ds.id2moa,
                "num_classes": train_ds.num_classes,
                "exclude_compounds": args.exclude_compounds,
                "test_metrics": test_metrics,
            }, save_path)
            print(f"  Saved best model (acc={best_acc:.4f}) -> {save_path}")

    # Save final
    save_path = os.path.join(args.checkpoint_dir, "final.pt")
    torch.save({
        "epoch": args.epochs,
        "classifier_state": model.state_dict_head(),
        "moa2id": train_ds.moa2id,
        "id2moa": train_ds.id2moa,
        "num_classes": train_ds.num_classes,
        "exclude_compounds": args.exclude_compounds,
        "test_metrics": evaluate(model, test_loader, device),
    }, save_path)
    print(f"Final model -> {save_path}")


if __name__ == "__main__":
    main()
