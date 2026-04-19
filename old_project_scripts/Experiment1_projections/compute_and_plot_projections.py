#!/usr/bin/env python3
"""Compute refusal projections per layer for text, image, and image+text modalities.

Projection formula (Eq. 5 from OmniSteer paper):
    p_l(x) = (h_l(x) - h̄_safe_text)^T v_refu_l / ||v_refu_l||²

Key corrections vs. original script:
    1. h̄_safe is computed ONCE from text-only safe samples and reused across all
       modalities (not recomputed per modality).
    2. Projection mode defaults to 'coefficient' (matching Eq. 5), not cosine.
       Cosine mode removes magnitude information, destroying the key signal.
    3. Pooling always uses the last token position (correct for causal LMs like
       LLaVA). Mean-pooling over all tokens dilutes the signal with image patch tokens.
    4. Per-layer min-max normalization is applied to produce 0-1 scaled plots
       comparable to Figure 5 of the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute projection of hidden states on refusal vector (paper-faithful)"
    )
    parser.add_argument("--model-id", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--refusal-file", type=Path, required=True,
                        help="Path to refusal_direction_*.pt produced by extract_refusal_direction.py")
    parser.add_argument("--text-jsonl", type=Path,
                        default=Path("dataset/extracted_text_image/text_only_prompts.jsonl"),
                        help="JSONL with text prompts (fields: text, type)")
    parser.add_argument("--image-text-jsonl", type=Path,
                        default=Path("dataset/extracted_text_image/image_text_pairs.jsonl"),
                        help="JSONL with image+text pairs (fields: text, image, type)")
    parser.add_argument("--image-root", type=Path,
                        default=Path("dataset/extracted_text_image"),
                        help="Root directory for resolving relative image paths")
    parser.add_argument("--harmful-label", type=str, default="harmful")
    parser.add_argument("--safe-label", type=str, default="safe")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max samples per class per modality")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-prefix", type=Path, default=None,
                        help="Output prefix for saved files. Defaults to refusal-file stem.")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Apply per-layer min-max normalization (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Skip normalization, plot raw coefficient values")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_by_label(items: List[dict], label: str, limit: int) -> List[dict]:
    picked = [x for x in items if x.get("type") == label]
    if not picked:
        raise ValueError(f"No items with label='{label}' found in dataset.")
    return picked[:limit]


# ---------------------------------------------------------------------------
# Model / hidden-state helpers
# ---------------------------------------------------------------------------

def extract_hidden_states(outputs) -> torch.Tensor:
    """Pull hidden_states tuple out of whatever the model returns."""
    hs = getattr(outputs, "hidden_states", None)
    if hs is None and hasattr(outputs, "language_model_outputs"):
        hs = getattr(outputs.language_model_outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError(
            "Could not find hidden_states in model outputs. "
            "Make sure output_hidden_states=True is supported by this model."
        )
    return torch.stack(hs, dim=0)  # [n_layers+1, B, T, D]


def pool_last_token(stacked: torch.Tensor, num_layers_expected: int | None) -> torch.Tensor:
    """
    Always use the LAST token position.

    For causal LMs (LLaVA, Qwen, etc.) the last token accumulates the full
    context. Mean-pooling is wrong here because:
      - It averages in all image patch tokens (for LLaVA, hundreds of them).
      - It washes out the semantic refusal signal.

    Args:
        stacked:             [n_layers(+1), B, T, D]
        num_layers_expected: if provided and stacked has an extra leading dim
                             (embedding layer), strip it.

    Returns:
        [n_layers, B, D]
    """
    if num_layers_expected is not None and stacked.shape[0] == num_layers_expected + 1:
        stacked = stacked[1:]  # drop embedding layer
    return stacked[:, :, -1, :]   # [L, B, D]


def build_messages(modality: str, text: str) -> List[dict]:
    """Build chat-template messages for a given modality."""
    if modality == "text":
        content = [{"type": "text", "text": text}]
    elif modality == "image":
        # Image-only: use a neutral prompt so the model processes the image
        # without a text payload — the harmful content lives in the image.
        content = [{"type": "image"}, {"type": "text", "text": "What does this image show?"}]
    elif modality == "image_text":
        content = [{"type": "image"}, {"type": "text", "text": text}]
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return [{"role": "user", "content": content}]


def get_pooled_hidden(
    model,
    processor,
    item: dict,
    modality: str,
    image_root: Path,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward a single sample and return last-token pooled hidden states.

    Returns:
        Tensor of shape [L, D], float32, on CPU.
    """
    text = item.get("text", "")
    messages = build_messages(modality, text)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = None
    if modality in {"image", "image_text"}:
        rel = item.get("image")
        if not rel:
            raise ValueError(f"item missing 'image' field for modality={modality}: {item}")
        image = Image.open(image_root / rel).convert("RGB")

    if image is not None:
        # truncation=False: never drop <image> placeholder tokens
        encoded = processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True, truncation=False,
        )
    else:
        encoded = processor(
            text=[prompt],
            return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        try:
            outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
        except TypeError:
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)

    stacked = extract_hidden_states(outputs)   # [L+1, 1, T, D]

    num_layers = getattr(
        getattr(model.config, "text_config", model.config),
        "num_hidden_layers", None
    )
    pooled = pool_last_token(stacked, num_layers_expected=num_layers)  # [L, 1, D]
    return pooled.squeeze(1).detach().cpu().float()  # [L, D]


# ---------------------------------------------------------------------------
# Global safe mean (FIX 1: computed once from text, reused everywhere)
# ---------------------------------------------------------------------------

def compute_global_safe_mean(
    model,
    processor,
    safe_items: List[dict],
    image_root: Path,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute h̄_safe as the mean hidden state over TEXT-ONLY safe samples.

    This is the fixed reference used to centre ALL modalities (Eq. 5).
    Computing a separate h̄_safe per modality breaks cross-modal comparability.

    Returns:
        [L, D] float32 CPU tensor.
    """
    print(f"  Computing global safe mean from {len(safe_items)} text safe samples...")
    hidden_list = []
    for i, item in enumerate(safe_items):
        h = get_pooled_hidden(model, processor, item, "text", image_root, max_length, device)
        hidden_list.append(h)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(safe_items)}")
    return torch.stack(hidden_list, dim=0).mean(dim=0)  # [L, D]


# ---------------------------------------------------------------------------
# Projection (FIX 2: coefficient mode = Eq. 5)
# ---------------------------------------------------------------------------

def project_onto_refusal(
    h_x: torch.Tensor,       # [L, D]
    h_safe: torch.Tensor,    # [L, D]  — global text safe mean
    v_refu: torch.Tensor,    # [L, D]  — refusal direction
) -> torch.Tensor:
    """
    Eq. 5:  p_l(x) = (h_l(x) - h̄_safe_l)^T v_refu_l / ||v_refu_l||²

    This is a scalar projection (signed coefficient along v_refu).
    It preserves magnitude information, which is exactly what the paper
    analyses as the primary driver of Mid-layer Dissolution.

    Returns:
        [L] float32 tensor of projection values.
    """
    diff = h_x - h_safe                                      # [L, D]
    dot = (diff * v_refu).sum(dim=1)                         # [L]
    v_norm_sq = v_refu.norm(dim=1).pow(2).clamp_min(1e-8)   # [L]
    return dot / v_norm_sq                                   # [L]


# ---------------------------------------------------------------------------
# Per-modality projection loop
# ---------------------------------------------------------------------------

def compute_projections_for_modality(
    model,
    processor,
    modality: str,
    items: List[dict],
    image_root: Path,
    v_refu: torch.Tensor,
    h_safe_global: torch.Tensor,
    harmful_label: str,
    safe_label: str,
    limit: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with keys 'harmful' and 'safe', each [N, L] numpy arrays.
    """
    harmful_items = select_by_label(items, harmful_label, limit)
    safe_items = select_by_label(items, safe_label, limit)

    print(f"  {modality}: {len(harmful_items)} harmful, {len(safe_items)} safe samples")

    def run_batch(item_list: List[dict], label: str) -> np.ndarray:
        projs = []
        for i, item in enumerate(item_list):
            h_x = get_pooled_hidden(model, processor, item, modality,
                                    image_root, max_length, device)
            p = project_onto_refusal(h_x, h_safe_global, v_refu)  # [L]
            projs.append(p.numpy())
            if (i + 1) % 10 == 0:
                print(f"    [{label}] {i+1}/{len(item_list)}")
        return np.asarray(projs)   # [N, L]

    return {
        "harmful": run_batch(harmful_items, "harmful"),
        "safe":    run_batch(safe_items,    "safe"),
    }


# ---------------------------------------------------------------------------
# Normalization (FIX 4: per-layer min-max to get 0-1 range like paper Fig. 5)
# ---------------------------------------------------------------------------

def normalize_projections(results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply per-layer min-max normalization across ALL modalities and classes.

    We fit min/max jointly so the scale is shared across modalities,
    making the three panels directly comparable (as in the paper).
    """
    # Gather all values across modalities to get a global min/max per layer
    all_vals = []
    for modality_data in results.values():
        all_vals.append(modality_data["harmful"])  # [N, L]
        all_vals.append(modality_data["safe"])      # [N, L]

    concat = np.concatenate(all_vals, axis=0)      # [total_N, L]
    layer_min = concat.min(axis=0, keepdims=True)  # [1, L]
    layer_max = concat.max(axis=0, keepdims=True)  # [1, L]
    denom = (layer_max - layer_min).clip(min=1e-8)

    normalized: Dict[str, Dict[str, np.ndarray]] = {}
    for modality, modality_data in results.items():
        normalized[modality] = {
            "harmful": (modality_data["harmful"] - layer_min) / denom,
            "safe":    (modality_data["safe"]    - layer_min) / denom,
        }
    return normalized


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_single_modality(
    ax: plt.Axes,
    harmful: np.ndarray,  # [N, L]
    safe: np.ndarray,     # [N, L]
    title: str,
    ylabel: str,
) -> None:
    layers = np.arange(1, harmful.shape[1] + 1)
    h_mean, h_std = harmful.mean(axis=0), harmful.std(axis=0)
    s_mean, s_std = safe.mean(axis=0), safe.std(axis=0)

    ax.plot(layers, h_mean, color="red",  label="Harmful", linewidth=1.5)
    ax.fill_between(layers, h_mean - h_std, h_mean + h_std, color="red",  alpha=0.2)
    ax.plot(layers, s_mean, color="blue", label="Safe",    linewidth=1.5)
    ax.fill_between(layers, s_mean - s_std, s_mean + s_std, color="blue", alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_combined(
    results: Dict[str, Dict[str, np.ndarray]],
    out_file: Path,
    normalized: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    order = [("text", "Text"), ("image", "Image"), ("image_text", "Image+Text")]
    ylabel = "Normalized Projection $p_l(x)$" if normalized else "Projection $p_l(x)$"

    for ax, (key, title) in zip(axes, order):
        _plot_single_modality(
            ax,
            results[key]["harmful"],
            results[key]["safe"],
            title=f"Refusal Projection - {title}",
            ylabel=ylabel,
        )

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"Saved combined plot: {out_file}")


def plot_individual(
    results: Dict[str, Dict[str, np.ndarray]],
    output_prefix: Path,
    normalized: bool,
) -> None:
    order = [("text", "Text"), ("image", "Image"), ("image_text", "Image+Text")]
    ylabel = "Normalized Projection $p_l(x)$" if normalized else "Projection $p_l(x)$"

    for key, title in order:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_single_modality(
            ax,
            results[key]["harmful"],
            results[key]["safe"],
            title=f"Refusal Projection - {title}",
            ylabel=ylabel,
        )
        out = Path(str(output_prefix) + f"_{key}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"Saved individual plot: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate paths
    for p, name in [
        (args.refusal_file,    "refusal-file"),
        (args.text_jsonl,      "text-jsonl"),
        (args.image_text_jsonl,"image-text-jsonl"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"[{name}] not found: {p}")

    # Load refusal direction
    refusal = torch.load(args.refusal_file, weights_only=False)
    v_refu: torch.Tensor = refusal["refusal_direction"].float()  # [L, D]

    ref_model_id = refusal.get("model_id")
    if ref_model_id and ref_model_id != args.model_id:
        raise ValueError(
            f"Model mismatch: refusal file was built for '{ref_model_id}' "
            f"but --model-id='{args.model_id}'"
        )

    device = torch.device(args.device)
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading model: {args.model_id} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(args.model_id)
    if getattr(processor, "chat_template", None) is None and hasattr(processor, "tokenizer"):
        processor.chat_template = getattr(processor.tokenizer, "chat_template", None)

    model = AutoModel.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # Read datasets
    text_items       = read_jsonl(args.text_jsonl)
    image_text_items = read_jsonl(args.image_text_jsonl)

    # ------------------------------------------------------------------
    # FIX 1: Compute the global safe mean ONCE from text-only safe samples.
    # This single reference is used for ALL modalities.
    # ------------------------------------------------------------------
    text_safe_items = select_by_label(text_items, args.safe_label, args.limit)
    print("\n=== Computing global h̄_safe (text modality, safe samples) ===")
    h_safe_global = compute_global_safe_mean(
        model, processor, text_safe_items,
        args.image_root, args.max_length, device,
    )
    print(f"  h̄_safe shape: {h_safe_global.shape}  (L={h_safe_global.shape[0]}, D={h_safe_global.shape[1]})\n")

    # Sanity check: v_refu and h_safe must have same layer count
    assert v_refu.shape[0] == h_safe_global.shape[0], (
        f"Layer count mismatch: v_refu has {v_refu.shape[0]} layers, "
        f"h_safe_global has {h_safe_global.shape[0]} layers. "
        "Make sure the refusal direction was extracted from the same model."
    )

    # ------------------------------------------------------------------
    # Compute projections for each modality
    # ------------------------------------------------------------------
    results: Dict[str, Dict[str, np.ndarray]] = {}

    modality_configs = [
        ("text",       text_items,       "Text"),
        ("image",      image_text_items, "Image"),
        ("image_text", image_text_items, "Image+Text"),
    ]

    for modality, items, label in modality_configs:
        print(f"=== Modality: {label} ===")
        results[modality] = compute_projections_for_modality(
            model=model,
            processor=processor,
            modality=modality,
            items=items,
            image_root=args.image_root,
            v_refu=v_refu,
            h_safe_global=h_safe_global,   # shared reference
            harmful_label=args.harmful_label,
            safe_label=args.safe_label,
            limit=args.limit,
            max_length=args.max_length,
            device=device,
        )
        print()

    # ------------------------------------------------------------------
    # FIX 4: Normalize to 0-1 for paper-comparable plots
    # ------------------------------------------------------------------
    plot_results = normalize_projections(results) if args.normalize else results

    # ------------------------------------------------------------------
    # Save + plot
    # ------------------------------------------------------------------
    if args.output_prefix is None:
        output_prefix = args.refusal_file.with_suffix("")
    else:
        output_prefix = args.output_prefix

    # Save raw + normalized projections
    save_path = Path(str(output_prefix) + "_projections.pt")
    torch.save({
        "model_id":        args.model_id,
        "refusal_file":    str(args.refusal_file),
        "normalized":      args.normalize,
        "raw_projections": {
            k: {"harmful": v["harmful"], "safe": v["safe"]}
            for k, v in results.items()
        },
        "plot_projections": {
            k: {"harmful": v["harmful"], "safe": v["safe"]}
            for k, v in plot_results.items()
        },
    }, save_path)
    print(f"Saved projections: {save_path}")

    combined_path = Path(str(output_prefix) + "_combined.png")
    plot_combined(plot_results, combined_path, normalized=args.normalize)
    plot_individual(plot_results, output_prefix, normalized=args.normalize)

    print("\nDone.")


if __name__ == "__main__":
    main()