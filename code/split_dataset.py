"""
Split COVA-X dataset into Train / Validation / Test sets.

Two-step usage (80/20 then 80/20 → 64% train, 16% val, 20% test):
    python split_dataset.py --two_step --seed 42

Single-step usage:
    python split_dataset.py --train 0.60 --val 0.20 --test 0.20 --seed 42
"""

import os, json, glob, shutil, argparse
from collections import defaultdict
import random

def load_all_conversations(dataset_dir):
    by_type = defaultdict(list)
    for scam_dir in glob.glob(os.path.join(dataset_dir, "*")):
        if not os.path.isdir(scam_dir):
            continue
        scam_type = os.path.basename(scam_dir)
        for jf in glob.glob(os.path.join(scam_dir, "*.json")):
            by_type[scam_type].append(jf)
    return by_type

def split_by_type(files, seed, two_step=False, train_pct=0.60, val_pct=0.20, test_pct=0.20):
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)
    n = len(files)

    if not two_step:
        n_train = int(round(n * train_pct))
        n_val = int(round(n * val_pct))
        n_test = n - n_train - n_val
        return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]
    else:
        n_train_temp = int(round(n * 0.80))
        train_temp = files[:n_train_temp]
        test = files[n_train_temp:]
        n_train = int(round(len(train_temp) * 0.80))
        return train_temp[:n_train], train_temp[n_train:], test

def copy_files(file_list, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for src_path in file_list:
        scam_type = os.path.basename(os.path.dirname(src_path))
        type_dir = os.path.join(dest_dir, scam_type)
        os.makedirs(type_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(type_dir, os.path.basename(src_path)))

def main():
    parser = argparse.ArgumentParser(description="Split COVA-X dataset")
    parser.add_argument("--two_step", action="store_true")
    parser.add_argument("--train", type=float, default=0.60)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the COVA-X_Dataset folder (scam-type subfolders)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder for train/val/test splits")
    args = parser.parse_args()

    if args.two_step:
        print(f"Two-step split: 80/20 → 80/20  (64/16/20), Seed={args.seed}")
        note = "Two-step: 80/20 then 80/20 → 64/16/20"
    else:
        total = args.train + args.val + args.test
        if abs(total - 1.0) > 1e-6:
            print(f"ERROR: splits must sum to 1.0, got {total}")
            return
        print(f"Single-step: Train={args.train:.0%}, Val={args.val:.0%}, Test={args.test:.0%}, Seed={args.seed}")
        note = f"Single-step: {args.train:.0%}/{args.val:.0%}/{args.test:.0%}"

    by_type = load_all_conversations(args.dataset_dir)
    print(f"Found {sum(len(v) for v in by_type.values())} conversations across {len(by_type)} types")

    train_all, val_all, test_all = [], [], []
    for scam_type, files in sorted(by_type.items()):
        tr, va, te = split_by_type(files, args.seed, args.two_step, args.train, args.val, args.test)
        train_all.extend(tr); val_all.extend(va); test_all.extend(te)
        print(f"  {scam_type:28s}: total={len(files):4d}  train={len(tr):4d}  val={len(va):4d}  test={len(te):4d}")

    copy_files(train_all, os.path.join(args.output_dir, "train"))
    copy_files(val_all, os.path.join(args.output_dir, "val"))
    copy_files(test_all, os.path.join(args.output_dir, "test"))

    for name, lst in [("train", train_all), ("val", val_all), ("test", test_all)]:
        with open(os.path.join(args.output_dir, f"{name}_manifest.json"), 'w') as f:
            json.dump({"split": name, "count": len(lst), "note": note, "files": [os.path.basename(p) for p in lst]}, f, indent=2)

    print(f"\nDone! Train={len(train_all)}  Val={len(val_all)}  Test={len(test_all)}")

if __name__ == "__main__":
    main()
