#!/usr/bin/env python3
"""Calculate per-layer refusal directions using ONLY text prompts.
Supports both LLaVA-1.5-7B and Qwen2.5-VL-7B models.

Refusal direction at layer l is computed as:
    v_refu_l = mean_harmful_l - mean_safe_l

where each sample's hidden activation at a layer is mean-pooled over non-padding tokens.

Usage:
    python3 compute_text_refusal_direction.py \
      --model-id llava-hf/llava-1.5-7b-hf \
      --output dataset/refusal_direction_llava7b_text.pt

    python3 compute_text_refusal_direction.py \
      --model-id Qwen/Qwen2.5-VL-7B-Instruct \
      --output dataset/refusal_direction_qwen2.5vl_text.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel, PreTrainedModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute refusal direction per layer (text-only)")
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=Path("dataset/extracted_text_image/text_only_prompts.jsonl"),
        help="Path to JSONL containing text prompts and a type field (harmful/safe)",
    )
    parser.add_argument(
        "--harmful-label",
        type=str,
        default="harmful",
        help="Label in the type field for the harmful class",
    )
    parser.add_argument(
        "--safe-label",
        type=str,
        default="safe",
        help="Label in the type field for the safe class",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face model id (e.g., llava-hf/llava-1.5-7b-hf or Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on texts loaded per class",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt file for the refusal direction",
    )
    return parser.parse_args()


def read_texts_by_label(
    jsonl_path: Path,
    harmful_label: str,
    safe_label: str,
    limit: int | None = None,
) -> Tuple[List[str], List[str]]:
    harmful_texts: List[str] = []
    safe_texts: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            label = row.get("type")
            if not (isinstance(text, str) and text.strip()):
                continue

            if label == harmful_label and (limit is None or len(harmful_texts) < limit):
                harmful_texts.append(text.strip())
            elif label == safe_label and (limit is None or len(safe_texts) < limit):
                safe_texts.append(text.strip())

            if limit is not None and len(harmful_texts) >= limit and len(safe_texts) >= limit:
                break

    if not harmful_texts:
        raise ValueError(f"No prompts found with type='{harmful_label}'")
    if not safe_texts:
        raise ValueError(f"No prompts found with type='{safe_label}'")

    return harmful_texts, safe_texts


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_text_backbone(model: PreTrainedModel) -> PreTrainedModel:
    """Return the internal text module for VLMs to pass text directly."""
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
        lm_outputs = outputs.language_model_outputs
        hidden_states = getattr(lm_outputs, "hidden_states", None)

    if hidden_states is None:
        raise RuntimeError("Could not access hidden states from model output")

    stacked = torch.stack(hidden_states, dim=0)  # [L_or_L+1, B, T, D]
    return stacked


def pooled_layer_activations(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [L, B, D] pooled hidden states for transformer layers only."""
    # Apply chat templates for safety prompts if the model requires it (like Qwen)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        formatted_texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            for t in texts
        ]
    elif "llava" in model.config.model_type.lower() or "llava" in getattr(model.config, "_name_or_path", "").lower():
        # Fallback LLaVA chat template injection to match projection script
        formatted_texts = [
            f"USER: {t}\nASSISTANT:"
            for t in texts
        ]
    else:
        formatted_texts = texts

    encoded = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    text_backbone = resolve_text_backbone(model)
    common_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
    }

    with torch.inference_mode():
        try:
            outputs = text_backbone(**common_kwargs)
        except TypeError:
            common_kwargs.pop("use_cache", None)
            outputs = text_backbone(**common_kwargs)

    stacked = extract_hidden_states(outputs)

    # Some models return embeddings as the first hidden state (L+1 items)
    # Estimate based on number of layers to drop embedding layer if present
    num_layers = None
    if hasattr(model.config, "text_config"):
        num_layers = getattr(model.config.text_config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(model.config, "num_hidden_layers", None)
    
    if num_layers is not None and stacked.shape[0] == num_layers + 1:
        stacked = stacked[1:]

    # Get the last valid token position for each sequence in the batch
    if tokenizer.padding_side == "left":
        last_tokens_idx = torch.full((attention_mask.shape[0],), attention_mask.shape[1] - 1, device=device)
    else:
        last_tokens_idx = attention_mask.sum(dim=1) - 1 # [B]
    
    last_tokens_idx = last_tokens_idx.unsqueeze(0).unsqueeze(-1).expand(stacked.shape[0], -1, stacked.shape[3]) # [L, B, D]
    
    # Gather the specific last token representation for each layer and batch
    pooled = stacked.gather(dim=2, index=last_tokens_idx.unsqueeze(2)).squeeze(2) # [L, B, D]
    return pooled.detach().cpu()


def compute_mean_layer_activation(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    running_sum = None
    total = 0

    for batch in batched(texts, batch_size):
        layer_batch = pooled_layer_activations(
            model=model,
            tokenizer=tokenizer,
            texts=batch,
            max_length=max_length,
            device=device,
        )  # [L, B, D]

        batch_sum = layer_batch.sum(dim=1).to(torch.float32)  # [L, D]
        running_sum = batch_sum if running_sum is None else running_sum + batch_sum
        total += layer_batch.shape[1]

    if running_sum is None or total == 0:
        raise ValueError("No samples were processed")

    return running_sum / total


def main() -> None:
    args = parse_args()

    if not args.prompts_jsonl.exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_jsonl}")

    harmful_texts, safe_texts = read_texts_by_label(
        args.prompts_jsonl,
        harmful_label=args.harmful_label,
        safe_label=args.safe_label,
        limit=args.limit,
    )

    print(f"Loaded harmful texts: {len(harmful_texts)}")
    print(f"Loaded safe texts:    {len(safe_texts)}")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading model {args.model_id} on {device} ({dtype})")
    
    # Use processor or tokenizer depending on what's available
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    print("Computing mean activations for harmful texts...")
    harmful_mean = compute_mean_layer_activation(
        model=model,
        tokenizer=tokenizer,
        texts=harmful_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    
    print("Computing mean activations for safe texts...")
    safe_mean = compute_mean_layer_activation(
        model=model,
        tokenizer=tokenizer,
        texts=safe_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    refusal_direction = harmful_mean - safe_mean  # [L, D]

    payload = {
        "model_id": args.model_id,
        "prompts_jsonl": str(args.prompts_jsonl),
        "harmful_label": args.harmful_label,
        "safe_label": args.safe_label,
        "num_harmful": len(harmful_texts),
        "num_safe": len(safe_texts),
        "refusal_direction": refusal_direction,
        "safe_mean": safe_mean,
        "harmful_mean": harmful_mean,
        "per_layer": {f"layer_{i+1}": refusal_direction[i] for i in range(refusal_direction.shape[0])},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"Saved refusal directions to: {args.output}")
    print(f"Shape: {tuple(refusal_direction.shape)}  # [num_layers, hidden_size]")


if __name__ == "__main__":
    main()
