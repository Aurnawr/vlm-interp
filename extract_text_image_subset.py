#!/usr/bin/env python3
"""Extract text+image pairs and text-only prompts from AdvBench-omni.

This script is designed for experiments that only use text and image modalities,
ignoring other modality combinations that may exist in the original dataset.

Outputs:
1) image_text_pairs.jsonl  -> rows that have both image and non-empty text
2) text_only_prompts.jsonl -> prompt-only rows (id/text/type/source metadata)
3) images/ (optional)      -> copied/serialized images used by #1

Example (local dataset folder):
    python3 extract_text_image_subset.py \
        --dataset-dir dataset \
        --output-dir dataset/extracted_text_image \
        --copy-images

Example (download only text+image files from HF):
    python3 extract_text_image_subset.py \
        --dataset-id ailor/AdvBench-omni \
        --output-dir extracted_ti \
        --copy-images
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from huggingface_hub import snapshot_download
from PIL import Image as PILImage


DEFAULT_ALLOWED_MODALITIES = ("image_text", "image", "text")
DEFAULT_JSONL_FILES = ("advbench_t2i.jsonl", "benign_t2i.jsonl", "advbench_i.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image+text pairs and text-only prompts from a multimodal dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Local dataset directory containing JSONL files and data/ images",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="ailor/AdvBench-omni",
        help="Hugging Face dataset id (used only when --dataset-dir is not provided)",
    )
    parser.add_argument(
        "--jsonl-files",
        nargs="+",
        default=list(DEFAULT_JSONL_FILES),
        help="JSONL files to process from dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/extracted_text_image"),
        help="Directory where extracted JSONL files (and optional images) are written",
    )
    parser.add_argument(
        "--allowed-modalities",
        type=str,
        nargs="+",
        default=list(DEFAULT_ALLOWED_MODALITIES),
        help="Only rows with modality in this list are considered",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="If set, writes image assets into <output-dir>/images",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of kept rows",
    )
    return parser.parse_args()


def iter_jsonl_rows(dataset_dir: Path, jsonl_files: List[str]) -> Iterable[Dict[str, Any]]:
    """Yield rows from selected JSONL files, annotating source file as split."""
    for jsonl_name in jsonl_files:
        jsonl_path = dataset_dir / jsonl_name
        if not jsonl_path.exists():
            continue

        split_name = jsonl_path.stem
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row.setdefault("split", split_name)
                row.setdefault("source_jsonl", jsonl_name)
                yield row


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in name)


def row_has_image_and_text(row: Dict[str, Any]) -> bool:
    text = row.get("text")
    image = row.get("image")
    has_text = isinstance(text, str) and text.strip() != ""
    has_image = image is not None
    return has_text and has_image


def save_image_field(image_field: Any, out_path: Path) -> Optional[Path]:
    """Serialize different HuggingFace image representations to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Case 1: already a path-like string
    if isinstance(image_field, str):
        src = Path(image_field)
        if src.exists() and src.is_file():
            shutil.copy2(src, out_path)
            return out_path
        return None

    # Case 2: HF Image-like dict with keys such as {'path':..., 'bytes':...}
    if isinstance(image_field, dict):
        maybe_path = image_field.get("path")
        maybe_bytes = image_field.get("bytes")

        if isinstance(maybe_path, str) and Path(maybe_path).exists():
            shutil.copy2(maybe_path, out_path)
            return out_path

        if isinstance(maybe_bytes, (bytes, bytearray)):
            with PILImage.open(io.BytesIO(maybe_bytes)) as img:
                img.save(out_path)
            return out_path

        return None

    # Case 3: decoded PIL image object
    if isinstance(image_field, PILImage.Image):
        image_field.save(out_path)
        return out_path

    return None


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"--dataset-dir does not exist: {dataset_dir}")
        return dataset_dir

    print(
        "No --dataset-dir provided; downloading only text+image JSONL files and image folders "
        f"from {args.dataset_id}"
    )
    downloaded_dir = snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        allow_patterns=[
            *args.jsonl_files,
            "data/adv_i/*",
            "data/adv_seg_t2i/*",
            "data/benign_seg_images/*",
        ],
    )
    return Path(downloaded_dir)


def extract_image_path(image_field: Any, dataset_dir: Path) -> Optional[str]:
    if isinstance(image_field, str):
        p = Path(image_field)
        if p.is_absolute() and p.exists():
            return str(p)
        candidate = (dataset_dir / p).resolve()
        if candidate.exists():
            return str(candidate)
        return str(p)

    if isinstance(image_field, dict):
        maybe_path = image_field.get("path")
        if isinstance(maybe_path, str):
            p = Path(maybe_path)
            if p.exists():
                return str(p)
            candidate = (dataset_dir / p).resolve()
            if candidate.exists():
                return str(candidate)
    return None


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_dataset_dir(args)
    print(f"Using dataset directory: {dataset_dir}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_pairs_path = out_dir / "image_text_pairs.jsonl"
    prompt_only_path = out_dir / "text_only_prompts.jsonl"
    images_dir = out_dir / "images"
    if args.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    allowed_modalities = {m.strip() for m in args.allowed_modalities}

    kept = 0
    skipped_modality = 0
    skipped_missing_fields = 0
    skipped_missing_image_file = 0

    with image_pairs_path.open("w", encoding="utf-8") as f_pairs, prompt_only_path.open(
        "w", encoding="utf-8"
    ) as f_prompts:
        for i, row in enumerate(iter_jsonl_rows(dataset_dir, args.jsonl_files)):
            modality = row.get("modality")
            if modality not in allowed_modalities:
                skipped_modality += 1
                continue

            if not row_has_image_and_text(row):
                skipped_missing_fields += 1
                continue

            sample_id = str(row.get("id", f"sample_{i}"))
            sample_type = row.get("type", None)
            text = row["text"].strip()

            # Write prompt-only record.
            prompt_record = {
                "id": sample_id,
                "text": text,
                "type": sample_type,
                "source_modality": modality,
                "split": row.get("split"),
            }
            f_prompts.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")

            # Write image+text pair record.
            pair_record: Dict[str, Any] = {
                "id": sample_id,
                "text": text,
                "type": sample_type,
                "modality": modality,
                "split": row.get("split"),
                "source_jsonl": row.get("source_jsonl"),
            }

            resolved_image_path = extract_image_path(row.get("image"), dataset_dir)
            if resolved_image_path is None:
                skipped_missing_image_file += 1
                continue

            if args.copy_images:
                image_name = sanitize_filename(sample_id) + ".png"
                target = images_dir / image_name
                saved = save_image_field(resolved_image_path, target)
                pair_record["image"] = str(saved.relative_to(out_dir)) if saved else None
                pair_record["image_saved"] = saved is not None
            else:
                # Keep canonical path relative to dataset root when possible.
                rp = Path(resolved_image_path)
                try:
                    pair_record["image"] = str(rp.relative_to(dataset_dir))
                except ValueError:
                    pair_record["image"] = str(rp)

            f_pairs.write(json.dumps(pair_record, ensure_ascii=False) + "\n")

            kept += 1
            if args.max_samples is not None and kept >= args.max_samples:
                break

    print("Extraction complete.")
    print(f"  Kept rows (image+text): {kept}")
    print(f"  Skipped by modality:    {skipped_modality}")
    print(f"  Skipped missing fields: {skipped_missing_fields}")
    print(f"  Skipped missing image:  {skipped_missing_image_file}")
    print(f"  Wrote: {image_pairs_path}")
    print(f"  Wrote: {prompt_only_path}")
    if args.copy_images:
        print(f"  Image directory: {images_dir}")


if __name__ == "__main__":
    main()
