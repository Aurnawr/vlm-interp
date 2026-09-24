import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Puts a scale on the text-vs-image refusal cosine:
#   - split-half reliability of v_text and v_image (the noise ceiling)
#   - bootstrap CIs on per-layer cosine and norms
#   - random-direction null for the low end
# Row i of every condition corresponds to prompt i (images are renderings of the
# same prompts), so text and image are always resampled / split with shared indices.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cache for model m lives at <m>-results/hidden_states_<m>.npz; outputs go to <m>-results/reliability/
MODEL_NAMES = {"llava": "LLaVA-1.5-7B", "qwen": "Qwen2.5-VL-7B"}


def cosine(a, b):
    # a, b: [..., layers, dim] -> [..., layers]; NaN where either vector is zero (layer 0)
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    # Resampled means leave ~1e-16 rounding residue on the zero layer; treat it as zero
    den = np.where(den < 1e-12, 0.0, den)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def weighted_means(X, W):
    # X: [n, layers, dim], W: [B, n] rows summing to 1 -> [B, layers, dim]
    n, L, D = X.shape
    return (W @ X.reshape(n, L * D)).reshape(len(W), L, D)


def refusal_vectors(hs, W_harmful, W_harmless):
    v_text = weighted_means(hs["harmful_text"], W_harmful) - weighted_means(hs["harmless_text"], W_harmless)
    v_image = weighted_means(hs["harmful_image"], W_harmful) - weighted_means(hs["harmless_image"], W_harmless)
    return v_text, v_image


def split_half(hs, n_harmful, n_harmless, n_splits, rng, chunk):
    """Random disjoint halves A/B of each prompt set (same split for both modalities).

    Returns per-split, per-layer cosines:
      text:  cos(v_text_A,  v_text_B)
      image: cos(v_image_A, v_image_B)
      cross: mean of cos(v_text_A, v_image_B) and cos(v_text_B, v_image_A)
             (cross-modal at the same sample size, on disjoint prompts)
    """
    def half_weights(n):
        WA, WB = np.zeros((n_splits, n)), np.zeros((n_splits, n))
        for s in range(n_splits):
            perm = rng.permutation(n)
            WA[s, perm[: n // 2]] = 1.0 / (n // 2)
            WB[s, perm[n // 2: 2 * (n // 2)]] = 1.0 / (n // 2)
        return WA, WB

    WA_h, WB_h = half_weights(n_harmful)
    WA_l, WB_l = half_weights(n_harmless)
    text, image, cross = [], [], []
    for i in range(0, n_splits, chunk):
        s = slice(i, i + chunk)
        tA, iA = refusal_vectors(hs, WA_h[s], WA_l[s])
        tB, iB = refusal_vectors(hs, WB_h[s], WB_l[s])
        text.append(cosine(tA, tB))
        image.append(cosine(iA, iB))
        cross.append(0.5 * (cosine(tA, iB) + cosine(tB, iA)))
    return np.concatenate(text), np.concatenate(image), np.concatenate(cross)


def bootstrap(hs, n_harmful, n_harmless, n_boot, rng, chunk):
    """Resample prompts with replacement; returns per-resample cosine and norms [B, layers]."""
    def boot_weights(n):
        idx = rng.integers(0, n, size=(n_boot, n))
        W = np.zeros((n_boot, n))
        for b in range(n_boot):
            W[b] = np.bincount(idx[b], minlength=n) / n
        return W

    W_h, W_l = boot_weights(n_harmful), boot_weights(n_harmless)
    cos, nt, ni = [], [], []
    for i in range(0, n_boot, chunk):
        s = slice(i, i + chunk)
        vt, vi = refusal_vectors(hs, W_h[s], W_l[s])
        cos.append(cosine(vt, vi))
        nt.append(np.linalg.norm(vt, axis=-1))
        ni.append(np.linalg.norm(vi, axis=-1))
    return np.concatenate(cos), np.concatenate(nt), np.concatenate(ni)


def random_null(v, n_random, rng):
    """|cos| between v (per layer) and random unit vectors -> [n_random, layers]."""
    R = rng.standard_normal((n_random, v.shape[-1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_unit = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.abs(R @ v_unit.T)


def ci(x, alpha=0.05):
    return np.nanpercentile(x, 100 * alpha / 2, axis=0), np.nanpercentile(x, 100 * (1 - alpha / 2), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_splits", type=int, default=200)
    parser.add_argument("--n_random", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=50, help="resamples per batched matmul (memory knob)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="llava")
    args = parser.parse_args()

    CACHE_PATH = os.path.join(ROOT, f"{args.model}-results", f"hidden_states_{args.model}.npz")
    OUT_DIR = os.path.join(ROOT, f"{args.model}-results", "reliability")
    model_name = MODEL_NAMES[args.model]

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"{CACHE_PATH} not found. Extract hidden states first "
                                "(tsne_plots_llava.py / extract_hidden_states_qwen.py).")
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading cached hidden states from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    hs = {k: cache[k].astype(np.float64) for k in
          ["harmful_text", "harmless_text", "harmful_image", "harmless_image"]}
    n_harmful, n_harmless = len(hs["harmful_text"]), len(hs["harmless_text"])
    assert len(hs["harmful_image"]) == n_harmful and len(hs["harmless_image"]) == n_harmless

    # Point estimates on the full set (same as cosine_similarity.py)
    ones = lambda n: np.full((1, n), 1.0 / n)
    v_text, v_image = (v[0] for v in refusal_vectors(hs, ones(n_harmful), ones(n_harmless)))
    cos_full = cosine(v_text, v_image)
    norm_text, norm_image = np.linalg.norm(v_text, axis=-1), np.linalg.norm(v_image, axis=-1)
    num_layers = len(cos_full)
    layers = np.arange(num_layers)

    print(f"Split-half reliability ({args.n_splits} random splits)...")
    sh_text, sh_image, sh_cross = split_half(hs, n_harmful, n_harmless, args.n_splits, rng, args.chunk)
    r_text, r_image, r_cross = (np.nanmean(x, axis=0) for x in (sh_text, sh_image, sh_cross))
    # Spearman-Brown: self-consistency of two independent full-size (n, not n/2) estimates
    sb = lambda r: 2 * r / (1 + r)
    r_text_full, r_image_full = sb(r_text), sb(r_image)
    # Disattenuated cross-modal cosine, all terms at the same (half) sample size:
    # ~1 means the directions agree up to sampling noise; <1 is a real difference
    with np.errstate(invalid="ignore"):
        cos_disatt = r_cross / np.sqrt(r_text * r_image)

    print(f"Bootstrap ({args.n_boot} resamples)...")
    b_cos, b_nt, b_ni = bootstrap(hs, n_harmful, n_harmless, args.n_boot, rng, args.chunk)
    cos_lo, cos_hi = ci(b_cos)
    nt_lo, nt_hi = ci(b_nt)
    ni_lo, ni_hi = ci(b_ni)

    print(f"Random-direction null ({args.n_random} unit vectors)...")
    null = random_null(v_text, args.n_random, rng)
    null_95 = np.nanpercentile(null, 95, axis=0)

    np.savez(os.path.join(OUT_DIR, "reliability.npz"),
             cos_full=cos_full, norm_text=norm_text, norm_image=norm_image,
             sh_text=sh_text, sh_image=sh_image, sh_cross=sh_cross,
             r_text_full=r_text_full, r_image_full=r_image_full, cos_disatt=cos_disatt,
             boot_cos=b_cos, boot_norm_text=b_nt, boot_norm_image=b_ni, null_abs_cos=null)

    # Table
    lines = [
        f"{model_name}  n_harmful={n_harmful} n_harmless={n_harmless} n_boot={args.n_boot} "
        f"n_splits={args.n_splits} n_random={args.n_random} seed={args.seed}",
        "cross = cos(v_text, v_image) on full set [95% bootstrap CI]",
        "text_sh / image_sh = split-half self-consistency at n/2; *_full = Spearman-Brown to n",
        "cross_sh = cross-modal cos at n/2 on disjoint halves; disatt = cross_sh / sqrt(text_sh*image_sh)",
        "null95 = 95th pct |cos(v_text, random unit vector)|",
        "",
        f"{'layer':>5} {'cross':>7} {'95% CI':>17} {'text_sh':>8} {'text_full':>9} "
        f"{'image_sh':>8} {'image_full':>10} {'cross_sh':>8} {'disatt':>7} {'null95':>7}",
    ]
    for l in layers:
        lines.append(
            f"{l:>5} {cos_full[l]:>7.3f} [{cos_lo[l]:>6.3f}, {cos_hi[l]:>6.3f}] "
            f"{r_text[l]:>8.3f} {r_text_full[l]:>9.3f} {r_image[l]:>8.3f} {r_image_full[l]:>10.3f} "
            f"{r_cross[l]:>8.3f} {cos_disatt[l]:>7.3f} {null_95[l]:>7.3f}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(OUT_DIR, "reliability_table.txt"), "w") as f:
        f.write(table + "\n")

    # Cosine with ceilings and null
    plt.figure(figsize=(8, 5))
    plt.fill_between(layers, cos_lo, cos_hi, alpha=0.25, color="purple")
    plt.plot(layers, cos_full, marker="s", color="purple", label="cos(v_text, v_image), 95% bootstrap CI")
    plt.plot(layers, r_text_full, "--", color="C0", label="text split-half ceiling (Spearman-Brown)")
    plt.plot(layers, r_image_full, "--", color="C1", label="image split-half ceiling (Spearman-Brown)")
    plt.plot(layers, null_95, ":", color="gray", label="random-direction null (95th pct |cos|)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.title(f"{model_name}: text vs image refusal cosine against noise ceiling and null")
    plt.ylim(-0.4, 1.05)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(OUT_DIR, "cosine_with_ceiling.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Disattenuated cosine
    plt.figure(figsize=(8, 5))
    plt.plot(layers, cos_disatt, marker="o", color="purple")
    plt.axhline(1.0, ls="--", color="gray")
    plt.xlabel("Layer")
    plt.ylabel("cross_sh / sqrt(text_sh * image_sh)")
    plt.title(f"{model_name}: noise-corrected text vs image refusal cosine (1 = same up to noise)")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "cosine_disattenuated.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Norms with CIs
    plt.figure(figsize=(8, 5))
    plt.fill_between(layers, nt_lo, nt_hi, alpha=0.25, color="C0")
    plt.plot(layers, norm_text, marker="o", color="C0", label="Text norm")
    plt.fill_between(layers, ni_lo, ni_hi, alpha=0.25, color="C1")
    plt.plot(layers, norm_image, marker="x", color="C1", label="Image norm")
    plt.xlabel("Layer")
    plt.ylabel("Norm")
    plt.title(f"{model_name}: norm of refusal vectors, 95% bootstrap CI")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "norms_with_ci.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Done! Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
