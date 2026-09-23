ablation1: image-only condition (the original setup, before the image+text change)

Image prompt:  "USER: <image>\nASSISTANT:"   (no text payload)
Text prompt:   "USER: {text}\nASSISTANT:"
Model: llava-hf/llava-1.5-7b-hf fp16, n = 400 per class, last-token hidden states.
Data: the original dataset2/ (not committed; regenerated later with make_dataset2.py,
      so a re-run will not reproduce these exactly). Missing images were silently
      replaced with blank white ones at the time.
PCA layers: 0, 8, 16, 24, 32.

Headline from this run: cos(v_text, v_image) ~0.30 at layer 32, near-orthogonal mid-stack;
image refusal vector ~half the norm of the text one.
