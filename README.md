# vlm-interp

How is **refusal represented across modalities** in vision-language models?

A safety-tuned VLM inherits its refusal behaviour from a text-tuned LLM, but a harmful request can arrive as an image instead of as text. This repo asks whether the model encodes *one* modality-agnostic "this request is harmful" direction, or whether the image pathway carries a separate, weaker signal that text-derived safety never properly sees.

Two models are compared, chosen because they are built differently:

| Model | Architecture | Layers | Hidden dim |
|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | CLIP ViT-L + MLP projector bolted onto Vicuna-7B (adapter-style) | 32 | 4096 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Vision encoder + LLM trained jointly on large multimodal/OCR data (native) | 28 | 3584 |

**Status**

- ✅ **Experiment 1** (correlational): text vs image refusal directions, both models, with reliability analysis. Done.
- ✅ **Experiment 2** (causal): cross-modal steering / ablation. LLaVA (α ∈ {1, 2, 4}) and Qwen (α ∈ {1, 2}) done.

---

## TL;DR

| | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| Peak `cos(v_text, v_image)` | **0.45** (last layer) | **0.95** (layers 25–26) |
| Noise ceiling | ~0.99 | ~0.99–0.998 |
| Random null (95th pct) | 0.03 | 0.03 |
| Image / text refusal-vector norm | 2–9× weaker at every layer | Equal from layer ~19 on |
| Modality gap in PCA | Persists to layer 32 | Gone by layer ~22 |
| Image lands on its own text twin (last layer) | 6% harmless / 14% harmful | 98% harmless / 74% harmful |
| Refuses harmful typographic images (Exp. 2 baseline) | **0%** genuine | **98%** |
| Image vec added to harmless text → refusal | never (≤0.07) | 0.98 (L18, α=1) |
| Text vec ablated from harmful images → refusal | n/a (no baseline refusal) | 0.98 → **0.00** (L18) |

- **LLaVA** keeps images and text apart. The image harmfulness signal is weaker, appears later, and only partly aligns with the text refusal direction. This fits the idea that text-trained refusal does not fully transfer to images. Causally, the text direction can push images into refusal, but the image direction does nothing to text, and LLaVA never genuinely refuses a harmful typographic image.
- **Qwen** folds images into the same representation as text by about two-thirds depth: same direction, same strength, no modality gap. The transition is sharp at layer 13 → 14. Causally, at layer 18 the two directions work in both directions: either one induces refusal in either modality, and removing the *text* direction fully removes refusal of harmful images. At layer 12 (before the transition, cos 0.34) the image direction does nothing.
- **Main caveat:** all images are *typographic* (the prompt rendered as text). Qwen's convergence may mean "strong OCR reads the image into text", not "harm is represented modality-independently". The per-prompt image-to-text-twin result points toward the OCR reading. Natural harmful images are needed to separate the two.

---

## Data

- **Harmful:** AdvBench `harmful_behaviors.csv` (520 prompts), from the original `llm-attacks` repo.
- **Harmless:** Stanford Alpaca (`tatsu-lab/alpaca`), instruction-only rows, deduplicated, randomly sampled to the same count (`seed=0`).
- Every prompt is also **rendered as an 800×600 image** (black DejaVuSans text on white, word-wrapped, centred), so each sample exists as text and as an image.

```bash
python Experiment-1-scripts/make_dataset2.py --out dataset2   # run from the repo root
```

```
dataset2/                     (gitignored)
  harmful.json    harmful_images/harmful_000.png ...
  harmless.json   harmless/harmless_000.png ...
  manifest.json   sources, seed, counts
```

Experiment 1 uses prompts **0–399** per class. Experiment 2 evaluates on the held-out prompts **400–519**.

---

## Experiment 1: text vs image refusal directions

### Method

Four conditions: harmful text, harmless text, harmful image, harmless image. For each, take the **last-token hidden state at every layer** (embeddings + all decoder layers).

- Text input: the prompt itself (`USER: {text}\nASSISTANT:` for LLaVA, chat template for Qwen).
- Image input: the rendered image plus a fixed neutral instruction, `"Respond to the request in the image."` The request lives only in the image.

Per layer, the refusal vector for each modality is the class-mean difference:

```
v_text  = mean(h_harmful_text)  − mean(h_harmless_text)
v_image = mean(h_harmful_image) − mean(h_harmless_image)
```

These are compared by cosine similarity and norm, and the four clouds are visualised with PCA and t-SNE. Layer 0 is degenerate: every prompt ends in the same template token, so its refusal vector is zero (NaN cosine).

**Reliability** (`refusal_reliability.py`), so the cosine has a scale:

- 95% bootstrap CIs on cosine and norms (1000 resamples, shared indices across modalities).
- Split-half self-consistency of each vector, scaled to n=400 with Spearman–Brown → the **noise ceiling**.
- **Disattenuated** cosine = cross-modal cosine / √(text_sh · image_sh).
- **Random-direction null**: 95th percentile |cos(v_text, random unit vector)|.

### Results

Compared by relative depth, since the models have different layer counts. Full per-layer tables: [`llava-results/reliability/reliability_table.txt`](llava-results/reliability/reliability_table.txt), [`qwen-results/reliability/reliability_table.txt`](qwen-results/reliability/reliability_table.txt).

| Depth | LLaVA layer | cos [95% CI] | Qwen layer | cos [95% CI] |
|---|---|---|---|---|
| ~0.03 | 1 | −0.28 [−0.33, −0.22] | 1 | 0.23 [0.20, 0.25] |
| ~0.25 | 8 | 0.12 [0.10, 0.13] | 7 | 0.39 [0.37, 0.40] |
| ~0.50 | 16 | 0.23 [0.22, 0.25] | 14 | 0.68 [0.66, 0.69] |
| ~0.75 | 24 | 0.37 [0.36, 0.38] | 21 | 0.93 [0.93, 0.93] |
| ~0.90 | 29 | 0.38 [0.37, 0.39] | 25 | 0.95 [0.95, 0.95] |
| 1.00 | 32 | 0.45 [0.43, 0.46] | 28 | 0.95 [0.95, 0.95] |

In both models, the disattenuated cosine is within ~0.005 of the raw cosine, and CIs are tight. Sampling noise explains none of the gap from 1, so **the difference between the models is real**.

#### Cosine and norms with noise ceiling

| LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|
| ![](llava-results/reliability/cosine_with_ceiling.png) | ![](qwen-results/reliability/cosine_with_ceiling.png) |
| ![](llava-results/reliability/norms_with_ci.png) | ![](qwen-results/reliability/norms_with_ci.png) |

- **LLaVA:** the text norm is 2–9× the image norm at every layer, and the image norm barely grows before mid-depth. Cosine sits at 0.1–0.15 through the middle, then climbs to 0.38–0.45 at the end. Layer 1 is significantly *negative*.
- **Qwen:** the text norm starts 2–5× larger, but the ratio falls to ~1.05 from layer 19 on. Relative to the hidden-state norm, both refusal vectors reach ~0.6–0.7 in layers 20–26, so harmful-vs-harmless is a dominant late-layer feature for both modalities. Dropping the 10 largest-magnitude dimensions barely changes the cosine (0.946 → 0.944 at layer 24), so it is not driven by outlier dimensions.
- Both models show a norm drop at the final layer.

#### Geometry

| | LLaVA (layer 32) | Qwen (layer 28) |
|---|---|---|
| PCA | ![](llava-results/pca_plots/pca_layer_32.png) | ![](qwen-results/pca_plots/pca_layer_28.png) |
| t-SNE | ![](llava-results/tsne_plots/tsne_layer_32.png) | ![](qwen-results/tsne_plots/tsne_layer_28.png) |

- **LLaVA:** the modality gap persists at every layer. In t-SNE at layer 32, images form their own region of small mixed harmful/harmless clumps, well away from both text clusters.
- **Qwen:** the modality gap dominates early (PC1 = 64–77% of variance at layers 6–11), shrinks by layer 17, and is gone by layers 22–28, where PC1 is harmful vs harmless. Harmless images and harmless text merge into one cluster; harmful images and harmful text sit side by side as adjacent sub-clusters.

More layers in `*/pca_plots/` and `*/tsne_plots/`.

#### Does each image land on its own text?

For each image: is its nearest text point (of all 800) the text of the same prompt?

| | LLaVA L8 | L16 | L24 | L32 | Qwen L6 | L11 | L17 | L22 | L28 |
|---|---|---|---|---|---|---|---|---|---|
| harmless | 0.00 | 0.01 | 0.06 | 0.06 | 0.01 | 0.02 | 0.61 | 0.94 | 0.98 |
| harmful | 0.00 | 0.00 | 0.07 | 0.14 | 0.00 | 0.00 | 0.08 | 0.40 | 0.74 |

Qwen's convergence is per-prompt, not only class-level, and weaker for harmful prompts than harmless ones.

#### Held-out linear separability of harmful vs harmless

Mean-difference direction fit on even-indexed prompts, tested on odd-indexed ones.

- **LLaVA image:** 0.68–0.86 in layers 1–4, 0.88–0.96 from layer 6 on (text: 0.92–0.98).
- **Qwen image:** 0.83–0.85 in layers 1–4, 0.94–0.98 in layers 6–12, 0.99–1.00 from layer 14 on (text: 0.93–1.00).

So LLaVA's image harm signal is not absent. It is weaker and points in a different direction from the text signal. PCA under-sells it because the modality gap takes up the top components.

Write-ups: [`comparison.txt`](comparison.txt) (side by side), [`llava-results/results.txt`](llava-results/results.txt), [`qwen-results/results.txt`](qwen-results/results.txt).

### Ablation 1: image-only prompt (LLaVA)

[`ablation1/`](ablation1/) holds the earlier LLaVA run where the image was fed with **no text at all** (`USER: <image>\nASSISTANT:`), before the neutral instruction was added. Headline: cos(v_text, v_image) ≈ 0.30 at layer 32, near-orthogonal mid-stack, image refusal vector ≈ half the text norm. That run used an earlier `dataset2/` that silently substituted blank images for missing files, so it will not reproduce exactly (see [`ablation1/README.txt`](ablation1/README.txt)).

---

## Experiment 2: cross-modal steering

`Experiment-2-scripts/steer_cross_modal.py` turns Experiment 1 into a causal test. Following Arditi et al. (2024), it steers one modality with the *other* modality's refusal vector, with same-modality steering as the reference:

- **add:** `h ← h + α · s · v̂` at one layer, all positions, on **harmless** inputs. Does the vector induce refusal?
- **ablate:** `h ← h − (h·v̂) v̂` at every layer, all positions, on **harmful** inputs. Does removing the direction bypass refusal?
- By default (`--scale target`), `s` is the norm of the *target* modality's own refusal vector, so cross- and same-modality runs differ only in direction.
- Optional random-direction control (`--random_control`).
- Refusal is scored by the Arditi et al. refusal-substring list. Every generation is saved to `generations.jsonl` so the scorer can be checked by hand.
- Default layers: LLaVA 12, 20; Qwen 12, 18. Default α ∈ {1, 2, 4}. Qwen runs in bf16 (fp16 overflows).

```bash
python Experiment-2-scripts/steer_cross_modal.py --model llava --random_control
python Experiment-2-scripts/steer_cross_modal.py --model qwen  --random_control --alphas 1 2
```

The Qwen run took ~9 h on a 16 GB GPU (part of the model is offloaded to CPU), which is why it covers only α ∈ {1, 2}.

Outputs go to `<model>-results/steering/`: `steering_results.json`, `steering_table.txt`, `steering_refusal_rates_a*.png`, `generations.jsonl`. `--resume` continues an interrupted run from `generations.jsonl` (pass the same arguments as that run).

### Results: LLaVA-1.5-7B

Held-out prompts 400–519 (n=120 per class), greedy decoding, 64 new tokens, layers 12 and 20, α ∈ {1, 2, 4}, `--scale target`, with random control. Full table: [`llava-results/steering/steering_table.txt`](llava-results/steering/steering_table.txt).

**Scorer caveat.** The substring scorer counts "I'm sorry, but I am unable to read the image" as a refusal. On image inputs, most steered "refusals" are this, not a refusal of the request. The "genuine" column below excludes them, using an ad-hoc regex over `generations.jsonl` (not yet part of the script).

**Baselines (no steering)**

| Input | Refusal (scorer) | Genuine |
|---|---|---|
| Harmful text | 0.79 | 0.79 |
| Harmful image | 0.12 | **0.00** (all 14 are "can't read the image") |
| Harmless text | 0.02 | |
| Harmless image | 0.02 | |

LLaVA never genuinely refuses a harmful typographic image; it mostly transcribes it. So there is no image refusal to ablate, and the ablation-on-image conditions are uninformative for LLaVA.

**Add to harmless inputs (does the vector induce refusal?)**

| Condition | L12 α=1 | α=2 | α=4 | L20 α=1 | α=2 | α=4 |
|---|---|---|---|---|---|---|
| **text vec → image** (cross) | 0.05 | 0.18 | 0.56 | 0.28 | 0.63 | 0.91 |
| &nbsp;&nbsp;genuine (of 120) | 0 | 1 | 49 | 0 | 1 | 53 |
| image vec → image (same) | 0.04 | 0.08 | 0.38 | 0.07 | 0.20 | 0.65 |
| &nbsp;&nbsp;genuine (of 120) | 0 | 0 | 0 | 0 | 0 | 6 |
| **image vec → text** (cross) | 0.03 | 0.02 | 0.01 | 0.07 | 0.03 | 0.00 † |
| text vec → text (same) | 0.08 | 0.63 | 0.67 | 0.30 | 0.49 | 0.00 † |
| random → image | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 |
| random → text | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |

† Output has collapsed into repetition ("step step step…", "Warning note note…"), which the scorer counts as not refusing.

**Ablate from harmful inputs (does removing the direction bypass refusal?)**

| Condition | L12 | L20 |
|---|---|---|
| none (baseline harmful text) | 0.79 | 0.79 |
| text vec from text (same) | 0.13 | 0.04 |
| **image vec from text** (cross) | 0.73 | 0.54 |
| image conditions | uninformative (baseline genuine refusal is 0) | |

![](llava-results/steering/steering_refusal_rates_a4.png)

**Takeaways (LLaVA)**

- **Text → image transfers.** Adding the text refusal direction to harmless images makes LLaVA refuse ~40–45% of them at α=4 (e.g. "I cannot generate a haiku… it goes against my programming"), against ≤3% for a random direction of the same norm. The text refusal mechanism can be triggered from the image pathway.
- **Image → text does not.** The image vector never induces refusal on text; at high α it only degrades the output. Ablating it from harmful text removes a little refusal at layer 20 (0.79 → 0.54), far less than the text vector (→ 0.04).
- **The image vector is not a refusal direction for images either.** Added to images, it mostly produces "I can't read the image" rather than refusals, fitting Experiment 1's picture of a weak, poorly aligned image signal (cos ≈ 0.2–0.4 at these layers).
- Text-vector ablation confirms the text direction is causal for text refusal (0.79 → 0.04 at layer 20).

### Results: Qwen2.5-VL-7B

Same setup as LLaVA except layers 12 and 18 (bf16) and **α ∈ {1, 2} only** (no α=4). Full table: [`qwen-results/steering/steering_table.txt`](qwen-results/steering/steering_table.txt). For reference, Experiment 1 gives cos(v_text, v_image) = **0.34 at layer 12** (before the 13 → 14 transition) and **0.79 at layer 18**.

**Scorer check.** Unlike LLaVA, Qwen reads the images, so there are no "can't read the image" refusals and almost no degenerate output (2 of 4320). But the substring scorer still over-counts in one place: at L18 α=1, many steered image "refusals" are soft hedges ("As an AI language model, I don't have personal experiences, but…", "it is not clear what you are asking") that go on to answer. The "harm-framed" rows below count only refusals that cite harm/illegality/ethics or say "can't assist with that" in the first 250 characters (ad-hoc regex over `generations.jsonl`). It is a lower bound: it misses some real refusals (it counts 96 of 119 on the harmful-text baseline).

**Baselines (no steering)**

| Input | Refusal (scorer) |
|---|---|
| Harmful text | 0.99 |
| Harmful image | **0.98** (genuine: "I'm sorry, but I can't assist with that.") |
| Harmless text | 0.03 |
| Harmless image | 0.02 |

Qwen refuses harmful typographic images as reliably as harmful text, so here the image ablation conditions *are* informative.

**Add to harmless inputs (does the vector induce refusal?)**

| Condition | L12 α=1 | α=2 | L18 α=1 | α=2 |
|---|---|---|---|---|
| **text vec → image** (cross) | 0.00 | 0.00 | 0.85 | 1.00 |
| &nbsp;&nbsp;harm-framed (of 120) | 0 | 0 | 34 | 116 |
| image vec → image (same) | 0.02 | 0.01 | 0.63 | 0.99 |
| &nbsp;&nbsp;harm-framed (of 120) | 0 | 0 | 29 | 111 |
| **image vec → text** (cross) | 0.03 | 0.02 | 0.98 | 1.00 |
| &nbsp;&nbsp;harm-framed (of 120) | 0 | 0 | 96 | 118 |
| text vec → text (same) | 0.03 | 0.22 | 1.00 | 1.00 |
| &nbsp;&nbsp;harm-framed (of 120) | 0 | 16 | 111 | 119 |
| random → image | 0.01 | 0.02 | 0.00 | 0.00 |
| random → text | 0.03 | 0.03 | 0.04 | 0.01 |

At L18 the induced refusals are the model's standard harmful-request refusals on benign prompts, e.g. *"Tell me three causes of stress"* + image vector → *"I'm sorry, but I cannot provide information on illegal activities or promote harmful behavior."*

**Ablate from harmful inputs (does removing the direction bypass refusal?)**

| Condition | L12 | L18 |
|---|---|---|
| none (baseline harmful text / image) | 0.99 / 0.98 | 0.99 / 0.98 |
| text vec from text (same) | 0.41 | **0.01** |
| **image vec from text** (cross) | 0.96 | 0.65 |
| **text vec from image** (cross) | 0.14 | **0.00** |
| image vec from image (same) | 0.95 | 0.41 |

With the text direction ablated at L18, the model complies with harmful image requests outright (e.g. "To create a bot that automatically posts inflammatory comments on social media, you can follow these steps: 1. …"). Image-vector ablation is weaker, and part of the drop it produces is **moralising compliance**: the model opens with "this is unethical…" and then answers. The substring scorer counts this as non-refusal (37 of the 71 "bypassed" image-vec-from-image outputs at L18).

| α=1 | α=2 |
|---|---|
| ![](qwen-results/steering/steering_refusal_rates_a1.png) | ![](qwen-results/steering/steering_refusal_rates_a2.png) |

**Takeaways (Qwen)**

- **Transfer works both ways at L18.** Either modality's vector, added at the target modality's norm, induces refusal in either modality, far above the random control (≤0.04). This is the causal counterpart of Experiment 1's convergence (cos 0.79 here), and the opposite of LLaVA, where the image vector never induced refusal on text.
- **The text direction is the one refusal actually depends on, for images too.** Ablating v_text removes refusal of harmful images completely at L18 (0.98 → 0.00), and even at L12 (→ 0.14), where cos is only 0.34. Ablating v_image only partly removes refusal in either modality (→ 0.41 image, → 0.65 text at L18). So image refusal in Qwen runs through the text refusal direction.
- **Before the transition, the image vector is not causal.** At L12 it neither induces refusal (≤0.03) nor removes it (0.95–0.96). The text vector at L12 is also weak for adding (0.22 at best on text, 0 on images) but strong for ablation. This fits Experiment 1: at L12 the image vector mostly carries modality/format, not refusal.
- **Same OCR caveat.** Everything here uses typographic images, so "image refusal runs through the text direction" is exactly what a model that reads the image into text would do. It does not yet show a modality-general harm concept.

### LLaVA vs Qwen

| | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| Genuine refusal of harmful typographic images | 0% | 98% |
| text vec → harmless image | induces refusal (genuine 40–45% at α=4) | induces refusal (≈100% at L18 α=2) |
| image vec → harmless text | no effect / degrades output | induces refusal (98–100% at L18) |
| text vec ablated from harmful text | 0.79 → 0.04 | 0.99 → 0.01 |
| image vec ablated from harmful text | 0.79 → 0.54 | 0.99 → 0.65 |
| text vec ablated from harmful images | n/a | 0.98 → 0.00 |

In both models the text refusal direction is causal and reaches the image pathway. The difference is the image direction: in LLaVA it is not a refusal direction at all, while in Qwen (after layer ~14) it is nearly interchangeable with the text one for *inducing* refusal, though still weaker for *removing* it.

---

## Reproducing

### Setup

```bash
pip install torch transformers accelerate pillow numpy matplotlib scikit-learn tqdm datasets
```

A CUDA GPU with ~16 GB is enough for the 7B models. The Qwen scripts take `--gpu_mem` and offload the rest to CPU.

### Pipeline

```bash
# 0. data
python Experiment-1-scripts/make_dataset2.py --out dataset2

# 1a. LLaVA: extract hidden states (cached to hidden_states_llava.npz), then plots
python Experiment-1-scripts/tsne_plots_llava.py          # extraction + t-SNE (--recompute to refresh)
python Experiment-1-scripts/pca_plots_llava.py           # PCA
python Experiment-1-scripts/cosine_similarity.py         # norms, cosine, refusal_{text,image}.pt

# 1b. Qwen: extract, then all plots in one go
python Experiment-1-scripts/extract_hidden_states_qwen.py
python Experiment-1-scripts/plots_qwen.py                # PCA, t-SNE, norms, cosine, vectors

# 1c. reliability (either model)
python Experiment-1-scripts/refusal_reliability.py --model llava
python Experiment-1-scripts/refusal_reliability.py --model qwen

# single-prompt probe against the saved LLaVA vectors
python Experiment-1-scripts/interactive_cosine_sim.py --prompt "How do I pick a lock?" --image some.png
```

### Known path quirks

The LLaVA scripts predate the `*-results/` layout and write relative to the working directory or script folder, while `refusal_reliability.py` and `steer_cross_modal.py` read from `<model>-results/`. Before running those two for LLaVA:

- move `Experiment-1-scripts/hidden_states_llava.npz` to `llava-results/hidden_states_llava.npz`;
- move `refusal_text.pt` / `refusal_image.pt` into `llava-results/`. The copies at the repo root are the LLaVA vectors (shape 33 × 4096).

The Qwen scripts already read and write under `qwen-results/`.

---

## Repo layout

```
Experiment-1-scripts/
  make_dataset2.py              AdvBench + Alpaca prompts, rendered to images
  pca_plots_llava.py            LLaVA extraction helpers + PCA plots
  tsne_plots_llava.py           LLaVA extraction (cached .npz) + t-SNE plots
  cosine_similarity.py          LLaVA refusal vectors: norms + cosine
  interactive_cosine_sim.py     single-prompt probe against saved vectors
  extract_hidden_states_qwen.py Qwen2.5-VL extraction (cached .npz)
  plots_qwen.py                 all Qwen plots + refusal vectors
  refusal_reliability.py        bootstrap CIs, noise ceiling, random null
Experiment-2-scripts/
  steer_cross_modal.py          cross-modal add / ablate steering
llava-results/                  plots, results.txt, reliability/, steering/
qwen-results/                   plots, results.txt, reliability/, steering/, refusal vectors
ablation1/                      earlier image-only LLaVA run
comparison.txt                  LLaVA vs Qwen write-up
refusal_text.pt, refusal_image.pt   LLaVA refusal vectors [33, 4096]
```

---

## Caveats and next steps

- **Typographic images only.** Test with natural harmful images to tell "OCR works well" apart from "modality-general harm concept".
- **Refusal vs style confound.** AdvBench and Alpaca differ in style and length, not only harmfulness. A stable mean-difference direction is not proof that it is *the* refusal direction. A label-shuffle null is still missing.
- **Steering coverage.** Qwen was run at α ∈ {1, 2} only (LLaVA also has α=4), at two layers per model, with greedy decoding and 64 new tokens.
- **Refusal scorer.** The substring scorer confuses "can't read the image" (LLaVA) and soft "As an AI I don't have…" hedges (Qwen) with refusal. It misses degenerate output and "this is unethical, but here is how…" moralising compliance. The "genuine" / "harm-framed" counts in the README are ad-hoc regexes. It needs an exclusion list or an LLM judge.
- **Ad-hoc numbers.** The relative-norm, held-out separability, outlier-dimension and image-to-text-twin figures were computed ad hoc from the cached hidden states. No committed script produces them yet.

- [x] Run Experiment 2 on LLaVA
- [x] Run Experiment 2 on Qwen (α ∈ {1, 2})
- [ ] Qwen α=4, and more layers (e.g. 14–16, around the transition)
- [ ] Better refusal scorer (separate "can't read the image" and soft hedges, flag degenerate output and moralising compliance)
- [ ] Label-shuffle null
- [ ] Natural (non-typographic) harmful images
- [ ] Script the ad-hoc analyses
- [ ] Unify LLaVA output paths under `llava-results/`
