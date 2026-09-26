import os
import sys
import json
import argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Behavioural baseline on the VLSU 2x2 (make_vlsu_dataset.py), before extracting hidden states:
# for each cell (SS, HS, SH, HH = image safety x text safety), generate an unsteered response
# to the natural image + prompt pair and score refusal. Tells us whether each model refuses
# because of the image, the text, or only their combination.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "Experiment-2-scripts"))
from steer_cross_modal import MODELS, is_refusal, generate_batch  # noqa: E402

CELLS = ["SS", "HS", "SH", "HH"]


def load_cell(data_dir, cell, max_side):
    with open(os.path.join(data_dir, f"{cell}.json")) as f:
        items = json.load(f)
    for it in items:
        img = Image.open(os.path.join(data_dir, it["image"])).convert("RGB")
        img.thumbnail((max_side, max_side), Image.BICUBIC)  # caps Qwen's visual-token count
        it["pil"] = img
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="llava")
    parser.add_argument("--data", default=os.path.join(REPO_ROOT, "dataset_vlsu"))
    parser.add_argument("--max_side", type=int, default=672)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gpu_mem", default="11GiB")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    out_dir = os.path.join(REPO_ROOT, f"{args.model}-results", "vlsu")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    model_cls = LlavaForConditionalGeneration if args.model == "llava" else Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(cfg["id"])
    processor.tokenizer.padding_side = "left"
    model = model_cls.from_pretrained(cfg["id"], dtype=cfg["dtype"], device_map="auto",
                                      max_memory={0: args.gpu_mem, "cpu": "96GiB"})
    model.eval()

    def fmt(text):
        if args.model == "llava":
            return f"USER: <image>\n{text}\nASSISTANT:"
        return processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)

    records = []
    for cell in CELLS:
        items = load_cell(args.data, cell, args.max_side)
        outs = []
        for i in tqdm(range(0, len(items), args.batch_size), desc=cell):
            batch = items[i:i + args.batch_size]
            outs += generate_batch(model, processor, [fmt(it["prompt"]) for it in batch],
                                   [it["pil"] for it in batch], args.max_new_tokens)
        for it, o in zip(items, outs):
            o = o.strip()
            records.append({"cell": cell, **{k: v for k, v in it.items() if k != "pil"},
                            "completion": o, "refusal": is_refusal(o)})
        print(f"  {cell}  refusal = {np.mean([r['refusal'] for r in records if r['cell'] == cell]):.3f}")

    with open(os.path.join(out_dir, "baseline_generations.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(records)
    lines = [f"{cfg['name']}  VLSU 2x2 behavioural baseline  n={len(df) // len(CELLS)} per cell  "
             f"max_side={args.max_side}  max_new_tokens={args.max_new_tokens}",
             "cell = image safety x text safety (S = safe, H = unsafe); refusal = substring scorer", "",
             f"{'cell':<6}{'n':>5}{'refusal':>9}   by VLSU combined grade (refusal / n)"]
    for cell in CELLS:
        s = df[df["cell"] == cell]
        by = "  ".join(f"{g}: {s[s.combined_grade == g].refusal.mean():.2f}/{(s.combined_grade == g).sum()}"
                       for g in ["safe", "borderline", "unsafe"] if (s.combined_grade == g).any())
        lines.append(f"{cell:<6}{len(s):>5}{s.refusal.mean():>9.2f}   {by}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(out_dir, "baseline_table.txt"), "w") as f:
        f.write(table + "\n")
    print(f"Done! Outputs in {out_dir}")


if __name__ == "__main__":
    main()
