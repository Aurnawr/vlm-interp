#!/usr/bin/env python3
"""Compute refusal projections per layer for text, image, and image+text modalities.

Projection formula (per layer l):
    cosine mode (default):
        p_l(x) = ((h_l(x) - mean_safe_l)^T v_refu_l) / (||h_l(x) - mean_safe_l|| * ||v_refu_l||)

    coefficient mode:
        p_l(x) = ((h_l(x) - mean_safe_l)^T v_refu_l) / ||v_refu_l||^2

The script:
1) loads a precomputed refusal direction file (.pt),
2) extracts pooled hidden states for each sample/modality,
3) computes per-layer projections for harmful/safe samples,
4) saves projections to .pt and creates per-modality plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute projection of hidden states on refusal vector")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model id")
    parser.add_argument("--refusal-file", type=Path, required=True, help="Path to refusal_direction_*.pt")
    parser.add_argument(
        "--text-jsonl",
        type=Path,
        default=Path("dataset/extracted_text_image/text_only_prompts.jsonl"),
        help="JSONL with text prompts and type labels",
    )
    parser.add_argument(
        "--image-text-jsonl",
        type=Path,
        default=Path("dataset/extracted_text_image/image_text_pairs.jsonl"),
        help="JSONL with image+text pairs and type labels",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("dataset/extracted_text_image"),
        help="Root path used to resolve relative image paths in JSONL",
    )
    parser.add_argument("--harmful-label", type=str, default="harmful")
    parser.add_argument("--safe-label", type=str, default="safe")
    parser.add_argument("--limit", type=int, default=50, help="Max samples per class per modality")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--projection-mode",
        type=str,
        default="cosine",
        choices=("cosine", "coefficient"),
        help=(
            "Projection normalization mode. "
            "'cosine' computes normalized similarity in [-1, 1], "
            "'coefficient' uses ((h-mean_safe)^T v)/||v||^2."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Optional output prefix. Default: <refusal_file_without_suffix>",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_text_backbone(model):
    for attr in ("language_model", "model", "text_model", "llm"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            return candidate
    get_lm = getattr(model, "get_language_model", None)
    if callable(get_lm):
        candidate = get_lm()
        if candidate is not None:
            return candidate
    return model


def extract_hidden_states(outputs) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and hasattr(outputs, "language_model_outputs"):
        hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Could not access hidden states from model outputs")
    return torch.stack(hidden_states, dim=0)  # [L_or_L+1, B, T, D]


def pool_hidden_states(
    stacked: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_layers_expected: int | None,
) -> torch.Tensor:
    """Return pooled [L, B, D] using attention mask when aligned, else fallback to last token."""
    if num_layers_expected is not None and stacked.shape[0] == num_layers_expected + 1:
        stacked = stacked[1:]

    # Preferred: mask-aware mean pooling if sequence dims align.
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] == stacked.shape[2]:
        mask = attention_mask.unsqueeze(0).unsqueeze(-1).to(stacked.dtype)  # [1, B, T, 1]
        pooled = (stacked * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1e-6)
        return pooled

    # Fallback: causal models often store strongest aggregate signal in last position.
    return stacked[:, :, -1, :]


def build_messages(modality: str, text: str) -> List[dict]:
    if modality == "text":
        content = [{"type": "text", "text": text}]
    elif modality == "image":
        content = [{"type": "image"}, {"type": "text", "text": "Describe this image."}]
    elif modality == "image_text":
        content = [{"type": "image"}, {"type": "text", "text": text}]
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return [{"role": "user", "content": content}]


def pooled_activation_single(
    model,
    processor,
    item: dict,
    modality: str,
    image_root: Path,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    text = item.get("text", "")
    messages = build_messages(modality, text)

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = None
    if modality in {"image", "image_text"}:
        rel = item.get("image")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"Missing image path for modality={modality}: {item}")
        image_path = image_root / rel
        image = Image.open(image_path).convert("RGB")

    if image is not None:
        # Important: avoid truncation that can drop <image> placeholders.
        encoded = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    else:
        encoded = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    kwargs = {
        **encoded,
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
    }

    with torch.inference_mode():
        try:
            outputs = model(**kwargs)
        except TypeError:
            kwargs.pop("use_cache", None)
            outputs = model(**kwargs)

    stacked = extract_hidden_states(outputs)

    num_layers = getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers", None)
    attention_mask = encoded.get("attention_mask")
    pooled = pool_hidden_states(stacked, attention_mask, num_layers_expected=num_layers)  # [L, 1, D]
    return pooled.squeeze(1).detach().cpu().float()  # [L, D]


def select_by_label(items: List[dict], label: str, limit: int) -> List[dict]:
    picked = [x for x in items if x.get("type") == label]
    return picked[:limit]


def compute_modality_projections(
    model,
    processor,
    modality: str,
    items: List[dict],
    image_root: Path,
    v_refu: torch.Tensor,
    harmful_label: str,
    safe_label: str,
    limit: int,
    max_length: int,
    device: torch.device,
    projection_mode: str,
) -> Dict[str, np.ndarray]:
    harmful_items = select_by_label(items, harmful_label, limit)
    safe_items = select_by_label(items, safe_label, limit)

    if not harmful_items or not safe_items:
        raise ValueError(
            f"Not enough samples for modality={modality}. "
            f"harmful={len(harmful_items)}, safe={len(safe_items)}"
        )

    safe_hidden = [
        pooled_activation_single(model, processor, item, modality, image_root, max_length, device)
        for item in safe_items
    ]
    h_safe = torch.stack(safe_hidden, dim=0).mean(dim=0)  # [L, D]

    v_norm = v_refu.norm(dim=1).clamp_min(1e-8)  # [L]
    v_norm_sq = (v_norm * v_norm).clamp_min(1e-8)  # [L]

    def project(diff: torch.Tensor) -> torch.Tensor:
        if projection_mode == "coefficient":
            return (diff * v_refu).sum(dim=1) / v_norm_sq

        # Cosine-normalized projection is robust to tiny ||v|| and keeps values bounded.
        diff_norm = diff.norm(dim=1).clamp_min(1e-8)
        return (diff * v_refu).sum(dim=1) / (diff_norm * v_norm)

    harmful_proj: List[np.ndarray] = []
    for item in harmful_items:
        h_x = pooled_activation_single(model, processor, item, modality, image_root, max_length, device)
        diff = h_x - h_safe
        p = project(diff)
        harmful_proj.append(p.numpy())

    safe_proj: List[np.ndarray] = []
    for item in safe_items:
        h_x = pooled_activation_single(model, processor, item, modality, image_root, max_length, device)
        diff = h_x - h_safe
        p = project(diff)
        safe_proj.append(p.numpy())

    return {
        "harmful": np.asarray(harmful_proj),  # [N, L]
        "safe": np.asarray(safe_proj),        # [N, L]
    }


def plot_modality(name: str, harmful: np.ndarray, safe: np.ndarray, out_file: Path) -> None:
    layers = np.arange(1, harmful.shape[1] + 1)
    h_mean, h_std = harmful.mean(axis=0), harmful.std(axis=0)
    s_mean, s_std = safe.mean(axis=0), safe.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(layers, h_mean, color="red", label="Harmful")
    plt.fill_between(layers, h_mean - h_std, h_mean + h_std, color="red", alpha=0.2)
    plt.plot(layers, s_mean, color="blue", label="Safe")
    plt.fill_between(layers, s_mean - s_std, s_mean + s_std, color="blue", alpha=0.2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Projection $p_l(x)$")
    plt.title(f"Refusal Projection - {name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_combined(results: Dict[str, Dict[str, np.ndarray]], out_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    order = [("text", "Text"), ("image", "Image"), ("image_text", "Image+Text")]

    for ax, (key, title) in zip(axes, order):
        harmful = results[key]["harmful"]
        safe = results[key]["safe"]
        layers = np.arange(1, harmful.shape[1] + 1)
        h_mean, h_std = harmful.mean(axis=0), harmful.std(axis=0)
        s_mean, s_std = safe.mean(axis=0), safe.std(axis=0)

        ax.plot(layers, h_mean, color="red", label="Harmful")
        ax.fill_between(layers, h_mean - h_std, h_mean + h_std, color="red", alpha=0.2)
        ax.plot(layers, s_mean, color="blue", label="Safe")
        ax.fill_between(layers, s_mean - s_std, s_mean + s_std, color="blue", alpha=0.2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Refusal Projection - {title}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Projection $p_l(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.refusal_file.exists():
        raise FileNotFoundError(f"Refusal file not found: {args.refusal_file}")
    if not args.text_jsonl.exists():
        raise FileNotFoundError(f"Missing text JSONL: {args.text_jsonl}")
    if not args.image_text_jsonl.exists():
        raise FileNotFoundError(f"Missing image+text JSONL: {args.image_text_jsonl}")

    refusal = torch.load(args.refusal_file, weights_only=False)
    v_refu = refusal["refusal_direction"].float()  # [L, D]
    ref_model_id = refusal.get("model_id")
    if ref_model_id and ref_model_id != args.model_id:
        raise ValueError(
            f"Model/refusal mismatch. refusal file model_id={ref_model_id}, requested model_id={args.model_id}"
        )

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading processor/model for {args.model_id} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(args.model_id)
    if getattr(processor, "chat_template", None) is None and hasattr(processor, "tokenizer"):
        # fallback to tokenizer chat template if available
        processor.chat_template = getattr(processor.tokenizer, "chat_template", None)

    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    text_items = read_jsonl(args.text_jsonl)
    image_text_items = read_jsonl(args.image_text_jsonl)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for modality, items in (
        ("text", text_items),
        ("image", image_text_items),
        ("image_text", image_text_items),
    ):
        print(f"Computing modality={modality} with up to {args.limit} samples/class...")
        results[modality] = compute_modality_projections(
            model=model,
            processor=processor,
            modality=modality,
            items=items,
            image_root=args.image_root,
            v_refu=v_refu,
            harmful_label=args.harmful_label,
            safe_label=args.safe_label,
            limit=args.limit,
            max_length=args.max_length,
            device=device,
            projection_mode=args.projection_mode,
        )

    if args.output_prefix is None:
        output_prefix = args.refusal_file.with_suffix("")
    else:
        output_prefix = args.output_prefix

    payload = {
        "model_id": args.model_id,
        "refusal_file": str(args.refusal_file),
        "text_jsonl": str(args.text_jsonl),
        "image_text_jsonl": str(args.image_text_jsonl),
        "limit": args.limit,
        "projection_mode": args.projection_mode,
        "projections": {
            k: {"harmful": v["harmful"], "safe": v["safe"]}
            for k, v in results.items()
        },
    }

    projections_file = Path(str(output_prefix) + "_projections.pt")
    torch.save(payload, projections_file)

    plot_modality("Text", results["text"]["harmful"], results["text"]["safe"], Path(str(output_prefix) + "_text.png"))
    plot_modality("Image", results["image"]["harmful"], results["image"]["safe"], Path(str(output_prefix) + "_image.png"))
    plot_modality(
        "Image+Text",
        results["image_text"]["harmful"],
        results["image_text"]["safe"],
        Path(str(output_prefix) + "_image_text.png"),
    )
    plot_combined(results, Path(str(output_prefix) + "_combined.png"))

    print(f"Saved projections tensor: {projections_file}")
    print(f"Saved plots:")
    print(f"  - {Path(str(output_prefix) + '_text.png')}")
    print(f"  - {Path(str(output_prefix) + '_image.png')}")
    print(f"  - {Path(str(output_prefix) + '_image_text.png')}")
    print(f"  - {Path(str(output_prefix) + '_combined.png')}")


if __name__ == "__main__":
    main()
