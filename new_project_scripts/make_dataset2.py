"""Build dataset2/: harmful (AdvBench) and harmless (Alpaca) prompts, each with a
matching rendered-text image, in the layout load_data() expects:

    dataset2/
      harmful.json        harmful_images/harmful_000.png ...
      harmless.json       harmless/harmless_000.png ...
      manifest.json       sources, seed, counts

Run from inside new_project_scripts/ (like the other scripts), or pass --out.
"""
import os
import csv
import io
import json
import random
import argparse
import textwrap
import urllib.request

from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

# walledai/AdvBench on the Hub is gated; this is the original public source.
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
ALPACA_ID = "tatsu-lab/alpaca"

IMG_W, IMG_H = 800, 600  # same size as the blank fallback in load_data
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise FileNotFoundError("No TrueType font found; install fonts-dejavu or edit FONT_CANDIDATES.")


def render_text_image(text, font_size=36, margin=50):
    """Black text on white, word-wrapped and vertically centred. Shrinks the font
    until the text fits so long prompts are never clipped."""
    img = Image.new("RGB", (IMG_W, IMG_H), "white")
    draw = ImageDraw.Draw(img)
    while True:
        font = load_font(font_size)
        avg_char_w = draw.textlength("abcdefghijklmnopqrstuvwxyz", font=font) / 26
        chars_per_line = max(10, int((IMG_W - 2 * margin) / avg_char_w))
        wrapped = textwrap.fill(text, width=chars_per_line)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=10)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if (w <= IMG_W - 2 * margin and h <= IMG_H - 2 * margin) or font_size <= 12:
            break
        font_size -= 2
    draw.multiline_text(((IMG_W - w) / 2, (IMG_H - h) / 2), wrapped, fill="black", font=font, spacing=10)
    return img


def get_harmful():
    with urllib.request.urlopen(ADVBENCH_URL) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    return [row["goal"].strip() for row in rows]


def get_harmless(n, seed):
    ds = load_dataset(ALPACA_ID, split="train")
    # Instruction-only rows, so the prompt is self-contained like an AdvBench goal.
    pool = sorted({x["instruction"].strip() for x in ds if not x["input"].strip()})
    rng = random.Random(seed)
    return rng.sample(pool, n)


def write_split(out_dir, prompts, json_name, img_subdir, prefix):
    with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    img_dir = os.path.join(out_dir, img_subdir)
    os.makedirs(img_dir, exist_ok=True)
    for i, text in enumerate(prompts):
        render_text_image(text).save(os.path.join(img_dir, f"{prefix}_{i:03d}.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate dataset2/ for new_project_scripts.")
    parser.add_argument("--out", default="dataset2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    harmful = get_harmful()
    harmless = get_harmless(len(harmful), args.seed)
    print(f"harmful: {len(harmful)}  harmless: {len(harmless)}")

    write_split(args.out, harmful, "harmful.json", "harmful_images", "harmful")
    write_split(args.out, harmless, "harmless.json", "harmless", "harmless")

    manifest = {
        "harmful_source": ADVBENCH_URL,
        "harmless_source": f"{ALPACA_ID} (train, input == '' only, deduplicated, random.Random({args.seed}).sample)",
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "image_size": [IMG_W, IMG_H],
        "image_style": "black DejaVuSans text on white, word-wrapped, centred",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
