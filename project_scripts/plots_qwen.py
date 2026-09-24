import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# All the LLaVA plots for Qwen2.5-VL-7B, from the cache written by
# extract_hidden_states_qwen.py: PCA, t-SNE, refusal-vector norms, text-vs-image
# cosine, and the refusal vectors themselves. Outputs go to qwen-results/.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "qwen-results")
CACHE_PATH = os.path.join(OUT_DIR, "hidden_states_qwen.npz")
MODEL_NAME = "Qwen2.5-VL-7B"


def scatter(X_2d, y, title, xlabel, ylabel, path):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_2d[y==0, 0], X_2d[y==0, 1], label='Harmful Text', alpha=0.7, color='red', marker='o')
    plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], label='Harmless Text', alpha=0.7, color='green', marker='o')
    plt.scatter(X_2d[y==2, 0], X_2d[y==2, 1], label='Harmful Image', alpha=0.7, color='darkred', marker='x')
    plt.scatter(X_2d[y==3, 0], X_2d[y==3, 1], label='Harmless Image', alpha=0.7, color='darkgreen', marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"{CACHE_PATH} not found. Run extract_hidden_states_qwen.py first.")

    print(f"Loading cached hidden states from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    hs_harmful_text = cache["harmful_text"]
    hs_harmless_text = cache["harmless_text"]
    hs_harmful_image = cache["harmful_image"]
    hs_harmless_image = cache["harmless_image"]

    num_layers = hs_harmful_text.shape[1]
    # 6 evenly spaced layers from the embedding layer to the final layer
    layers_to_plot = np.linspace(0, num_layers - 1, 6).round().astype(int).tolist()

    pca_dir = os.path.join(OUT_DIR, "pca_plots")
    tsne_dir = os.path.join(OUT_DIR, "tsne_plots")
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)

    print(f"Generating PCA and t-SNE plots for layers: {layers_to_plot}")
    for layer_idx in layers_to_plot:
        X = np.vstack([hs_harmful_text[:, layer_idx, :], hs_harmless_text[:, layer_idx, :],
                       hs_harmful_image[:, layer_idx, :], hs_harmless_image[:, layer_idx, :]])
        y = np.array([0]*len(hs_harmful_text) + [1]*len(hs_harmless_text) +
                     [2]*len(hs_harmful_image) + [3]*len(hs_harmless_image))

        # Layer 0 is the raw embedding of the shared last token, so all points coincide
        if X.var(axis=0).sum() < 1e-8:
            print(f"Skipping layer {layer_idx}: all hidden states are identical")
            continue

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter(X_pca, y, f"{MODEL_NAME} Hidden States PCA (Layer {layer_idx})",
                f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                os.path.join(pca_dir, f"pca_layer_{layer_idx}.png"))

        # Reduce to 50-d with PCA first (standard t-SNE preprocessing)
        n_pca = min(50, X.shape[0], X.shape[1])
        X_reduced = PCA(n_components=n_pca).fit_transform(X)
        perplexity = min(args.perplexity, (X.shape[0] - 1) / 3)
        X_tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(X_reduced)
        scatter(X_tsne, y, f"{MODEL_NAME} Hidden States t-SNE (Layer {layer_idx}, perplexity={perplexity:g})",
                "t-SNE 1", "t-SNE 2", os.path.join(tsne_dir, f"tsne_layer_{layer_idx}.png"))

    # Refusal vectors, norms and cosine (as in cosine_similarity.py)
    refusal_text = hs_harmful_text.mean(axis=0) - hs_harmless_text.mean(axis=0)
    refusal_image = hs_harmful_image.mean(axis=0) - hs_harmless_image.mean(axis=0)
    norms_text = np.linalg.norm(refusal_text, axis=1)
    norms_image = np.linalg.norm(refusal_image, axis=1)
    # Layer 0 refusal vectors are zero -> cosine undefined (NaN)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine_sim = np.sum(refusal_text * refusal_image, axis=1) / (norms_text * norms_image)
    layers = np.arange(num_layers)

    torch.save(refusal_text, os.path.join(OUT_DIR, "refusal_text.pt"))
    torch.save(refusal_image, os.path.join(OUT_DIR, "refusal_image.pt"))

    plt.figure()
    plt.plot(layers, norms_text, label='Text Norm', marker='o')
    plt.plot(layers, norms_image, label='Image Norm', marker='x')
    plt.xlabel('Layer')
    plt.ylabel('Norm')
    plt.title(f'Norm of Refusal Vectors (Text vs Image), {MODEL_NAME}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'norms.png'))
    plt.close()

    plt.figure()
    plt.plot(layers, cosine_sim, label='Cosine Similarity', marker='s', color='purple')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity between Text and Image Refusal Vectors, {MODEL_NAME}')
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'cosine_similarity_plot.png'))
    plt.close()

    print(f"{'layer':>5} {'norm_text':>10} {'norm_image':>10} {'cosine':>7}")
    for l in layers:
        print(f"{l:>5} {norms_text[l]:>10.3f} {norms_image[l]:>10.3f} {cosine_sim[l]:>7.3f}")
    print(f"Done! Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
