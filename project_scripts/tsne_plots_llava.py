import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoProcessor, LlavaForConditionalGeneration

from pca_plots_llava import load_data, get_hidden_states_text, get_hidden_states_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_PATH = os.path.join(SCRIPT_DIR, "hidden_states_llava.npz")

def extract_hidden_states(num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    data_dir = os.path.join(REPO_ROOT, "dataset2")

    harmful_prompts, harmful_images = load_data(
        os.path.join(data_dir, "harmful.json"), os.path.join(data_dir, "harmful_images"),
        "harmful", max_samples=num_samples)
    harmless_prompts, harmless_images = load_data(
        os.path.join(data_dir, "harmless.json"), os.path.join(data_dir, "harmless"),
        "harmless", max_samples=num_samples)

    print("Extracting hidden states for harmful texts...")
    hs_harmful_text = get_hidden_states_text(model, processor, harmful_prompts, device)

    print("Extracting hidden states for harmless texts...")
    hs_harmless_text = get_hidden_states_text(model, processor, harmless_prompts, device)

    print("Extracting hidden states for harmful images...")
    hs_harmful_image = get_hidden_states_image(model, processor, harmful_images, device)

    print("Extracting hidden states for harmless images...")
    hs_harmless_image = get_hidden_states_image(model, processor, harmless_images, device)

    return hs_harmful_text, hs_harmless_text, hs_harmful_image, hs_harmless_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--recompute", action="store_true",
                        help="Re-extract hidden states even if the cache exists")
    args = parser.parse_args()

    # Cache hidden states so t-SNE settings can be tweaked without rerunning the model
    if os.path.exists(CACHE_PATH) and not args.recompute:
        print(f"Loading cached hidden states from {CACHE_PATH}")
        cache = np.load(CACHE_PATH)
        hs_harmful_text = cache["harmful_text"]
        hs_harmless_text = cache["harmless_text"]
        hs_harmful_image = cache["harmful_image"]
        hs_harmless_image = cache["harmless_image"]
    else:
        hs_harmful_text, hs_harmless_text, hs_harmful_image, hs_harmless_image = extract_hidden_states(args.num_samples)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH,
                 harmful_text=hs_harmful_text, harmless_text=hs_harmless_text,
                 harmful_image=hs_harmful_image, harmless_image=hs_harmless_image)
        print(f"Saved hidden states to {CACHE_PATH}")

    num_layers = hs_harmful_text.shape[1]

    output_dir = "tsne_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Same layers as the PCA plots so the two can be compared side by side
    layers_to_plot = np.linspace(0, num_layers - 1, 6).round().astype(int).tolist()

    print(f"Generating t-SNE plots for layers: {layers_to_plot}")
    for layer_idx in layers_to_plot:
        X_harmful_text = hs_harmful_text[:, layer_idx, :]
        X_harmless_text = hs_harmless_text[:, layer_idx, :]
        X_harmful_image = hs_harmful_image[:, layer_idx, :]
        X_harmless_image = hs_harmless_image[:, layer_idx, :]

        X = np.vstack([X_harmful_text, X_harmless_text, X_harmful_image, X_harmless_image])
        y = np.array(
            [0]*len(X_harmful_text) +
            [1]*len(X_harmless_text) +
            [2]*len(X_harmful_image) +
            [3]*len(X_harmless_image)
        )

        # Layer 0 is the raw embedding of the last token, which is the same ":" for every
        # prompt, so all points coincide and PCA/t-SNE would divide by zero
        if X.var(axis=0).sum() < 1e-8:
            print(f"Skipping layer {layer_idx}: all hidden states are identical")
            continue

        # Reduce 4096-d hidden states to 50-d with PCA first (standard t-SNE preprocessing)
        n_pca = min(50, X.shape[0], X.shape[1])
        X_reduced = PCA(n_components=n_pca).fit_transform(X)

        perplexity = min(args.perplexity, (X.shape[0] - 1) / 3)
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0)
        X_tsne = tsne.fit_transform(X_reduced)

        plt.figure(figsize=(10, 7))
        plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='Harmful Text', alpha=0.7, color='red', marker='o')
        plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='Harmless Text', alpha=0.7, color='green', marker='o')
        plt.scatter(X_tsne[y==2, 0], X_tsne[y==2, 1], label='Harmful Image', alpha=0.7, color='darkred', marker='x')
        plt.scatter(X_tsne[y==3, 0], X_tsne[y==3, 1], label='Harmless Image', alpha=0.7, color='darkgreen', marker='x')

        plt.title(f"LLaVA-7B Hidden States t-SNE (Layer {layer_idx}, perplexity={perplexity:g})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(alpha=0.3)

        plot_path = os.path.join(output_dir, f"tsne_layer_{layer_idx}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

if __name__ == "__main__":
    main()
