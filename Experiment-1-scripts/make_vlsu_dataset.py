import os
import io
import json
import argparse
import concurrent.futures
import pandas as pd
import requests
from PIL import Image

# 2x2 image-safety x text-safety dataset from VLSU (apple/ml-vlsu, data/VLSU.csv):
#   SS = safe image,   safe text        HS = unsafe image, safe text
#   SH = safe image,   unsafe text      HH = unsafe image, unsafe text
# using VLSU's image_grade and consensus_text_grade (borderline / not_sure are left out).
# Images are real web images, fetched from web_path; dead links are skipped, so each cell is
# drawn from a shuffled pool until --per_cell images download.
# Sexual (C6, C7), human-exploitation (C15) and jailbreak (C13) items are excluded if ANY of
# the image / text / combined category fields names them.

CELLS = {"SS": ("safe", "safe"), "HS": ("unsafe", "safe"), "SH": ("safe", "unsafe"), "HH": ("unsafe", "unsafe")}
EXCLUDE = ("C6:", "C7:", "C13:", "C15:")
CATEGORY_COLS = ["image_category", "text_category", "combined_category"]
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36"}


def fetch(row, img_dir):
    path = os.path.join(img_dir, f"{row['uuid']}.png")
    if os.path.exists(path):
        return path
    try:
        r = requests.get(row["web_path"], headers=HEADERS, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < 64:
            return None
        img.save(path)
        return path
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to ml-vlsu/data/VLSU.csv")
    parser.add_argument("--out", default="dataset_vlsu")
    parser.add_argument("--per_cell", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    excluded = df[CATEGORY_COLS].fillna("").apply(lambda r: any(v.startswith(EXCLUDE) for v in r), axis=1)
    df = df[~excluded]
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    manifest = {"source": "apple/ml-vlsu data/VLSU.csv", "seed": args.seed,
                "excluded_categories": list(EXCLUDE), "cells": {}}
    for cell, (ig, tg) in CELLS.items():
        pool = df[(df["image_grade"] == ig) & (df["consensus_text_grade"] == tg)]
        pool = pool.sample(frac=1, random_state=args.seed).to_dict("records")
        kept, tried = [], 0
        with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
            while len(kept) < args.per_cell and tried < len(pool):
                chunk = pool[tried:tried + args.workers * 2]
                tried += len(chunk)
                for row, path in zip(chunk, ex.map(lambda r: fetch(r, img_dir), chunk)):
                    if path and len(kept) < args.per_cell:
                        kept.append({"uuid": row["uuid"], "prompt": row["prompt"],
                                     "image": os.path.relpath(path, args.out),
                                     "image_grade": ig, "text_grade": tg,
                                     "combined_grade": row["consensus_combined_grade"],
                                     "image_category": row["image_category"] if isinstance(row["image_category"], str) else "",
                                     "text_category": row["text_category"] if isinstance(row["text_category"], str) else "",
                                     "combined_category": row["combined_category"] if isinstance(row["combined_category"], str) else ""})
        manifest["cells"][cell] = {"n": len(kept), "tried": tried, "pool": len(pool)}
        with open(os.path.join(args.out, f"{cell}.json"), "w") as f:
            json.dump(kept, f, indent=1)
        print(f"{cell}: kept {len(kept)} of {tried} tried (pool {len(pool)})")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
