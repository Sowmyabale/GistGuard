"""
Split COVA-X dataset into Train / Validation / Test sets.

Single-step usage (specify exact fractions):
    python split_cova_dataset.py --train 0.60 --val 0.20 --test 0.20 --seed 42

Two-step usage (80/20 then 80/20  →  64% train, 16% val, 20% test):
    python split_cova_dataset.py --two_step --seed 42
"""

import os
import json
import glob
import shutil
import argparse
from collections import defaultdict
import random

def load_all_conversations(dataset_dir):
    """Load all COVA-X JSON files grouped by scam_type."""
    by_type = defaultdict(list)
    scam_dirs = glob.glob(os.path.join(dataset_dir, "*"))
    for scam_dir in scam_dirs:
        if not os.path.isdir(scam_dir):
            continue
        scam_type = os.path.basename(scam_dir)
        json_files = glob.glob(os.path.join(scam_dir, "*.json"))
        for jf in json_files:
            by_type[scam_type].append(jf)
    return by_type

def split_by_type(file_list, seed, two_step=False, train_pct=0.60, val_pct=0.20, test_pct=0.20):
    """Stratified split for one scam type."""
    random.seed(seed)
    files = file_list.copy()
    random.shuffle(files)
    n = len(files)

    if not two_step:
        # Single-step: direct train/val/test split
        n_train = int(round(n * train_pct))
        n_val = int(round(n * val_pct))
        n_test = n - n_train - n_val
        return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]
    else:
        # Two-step:
        #   Step 1: 80% train_temp / 20% test
        #   Step 2: split train_temp into 80% train / 20% val
        n_train_temp = int(round(n * 0.80))
        train_temp = files[:n_train_temp]
        test = files[n_train_temp:]

        n_train = int(round(len(train_temp) * 0.80))
        train = train_temp[:n_train]
        val = train_temp[n_train:]
        return train, val, test

def copy_files(file_list, dest_dir):
    """Copy files to destination, preserving subdirectory name."""
    os.makedirs(dest_dir, exist_ok=True)
    for src_path in file_list:
        fname = os.path.basename(src_path)
        scam_type = os.path.basename(os.path.dirname(src_path))
        type_dir = os.path.join(dest_dir, scam_type)
        os.makedirs(type_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(type_dir, fname))

def write_manifest(split_name, file_list, out_dir, note=""):
    """Write a JSON manifest with metadata."""
    manifest = {
        "split": split_name,
        "count": len(file_list),
        "note": note,
        "files": [os.path.basename(p) for p in file_list],
    }
    path = os.path.join(out_dir, f"{split_name}_manifest.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Split COVA-X dataset")
    parser.add_argument("--dataset_dir", type=str,
                        default="dataset/COVA-X Dataset/COVA-X_Dataset",
                        help="Path to COVA-X_Dataset folder")
    parser.add_argument("--output_dir", type=str,
                        default="dataset/COVA-X Dataset/COVA-X_Split",
                        help="Where to write train/val/test folders")
    parser.add_argument("--two_step", action="store_true",
                        help="Two-step split: 80/20 then 80/20 → 64/16/20")
    parser.add_argument("--train", type=float, default=0.60, help="Train fraction (single-step only)")
    parser.add_argument("--val",   type=float, default=0.20, help="Validation fraction (single-step only)")
    parser.add_argument("--test",  type=float, default=0.20, help="Test fraction (single-step only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.two_step:
        print(f"Two-step split: 80/20 → 80/20  (effective: 64% train, 16% val, 20% test), Seed={args.seed}")
        note = "Two-step: 80/20 then 80/20 → 64/16/20"
    else:
        total = args.train + args.val + args.test
        if abs(total - 1.0) > 1e-6:
            print(f"ERROR: train ({args.train}) + val ({args.val}) + test ({args.test}) = {total}, must equal 1.0")
            return
        print(f"Single-step split: Train={args.train:.0%}, Val={args.val:.0%}, Test={args.test:.0%}, Seed={args.seed}")
        note = f"Single-step: {args.train:.0%}/{args.val:.0%}/{args.test:.0%}"

    by_type = load_all_conversations(args.dataset_dir)
    print(f"Found {sum(len(v) for v in by_type.values())} conversations across {len(by_type)} scam types")

    train_all, val_all, test_all = [], [], []
    for scam_type, files in sorted(by_type.items()):
        tr, va, te = split_by_type(
            files, args.seed,
            two_step=args.two_step,
            train_pct=args.train, val_pct=args.val, test_pct=args.test
        )
        train_all.extend(tr)
        val_all.extend(va)
        test_all.extend(te)
        print(f"  {scam_type:28s}: total={len(files):4d}  train={len(tr):4d}  val={len(va):4d}  test={len(te):4d}")

    # Copy files
    copy_files(train_all, os.path.join(args.output_dir, "train"))
    copy_files(val_all,   os.path.join(args.output_dir, "val"))
    copy_files(test_all,  os.path.join(args.output_dir, "test"))

    # Write manifests
    write_manifest("train", train_all, args.output_dir, note)
    write_manifest("val",   val_all,   args.output_dir, note)
    write_manifest("test",  test_all,  args.output_dir, note)

    print(f"\nDone. Splits saved to: {args.output_dir}")
    print(f"  Train : {len(train_all)} files")
    print(f"  Val   : {len(val_all)} files")
    print(f"  Test  : {len(test_all)} files")

if __name__ == "__main__":
    main()
