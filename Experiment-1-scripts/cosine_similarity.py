import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Hidden states cached by tsne_plots_llava.py: [num_samples, num_layers, hidden_dim] per condition
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hidden_states_llava.npz")

def main():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"{CACHE_PATH} not found. Run tsne_plots_llava.py first to extract hidden states.")

    print(f"Loading cached hidden states from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    hs_harmful_text = cache["harmful_text"]
    hs_harmless_text = cache["harmless_text"]
    hs_harmful_image = cache["harmful_image"]
    hs_harmless_image = cache["harmless_image"]

    h_harmful_text_mean = hs_harmful_text.mean(axis=0)
    h_harmless_text_mean = hs_harmless_text.mean(axis=0)

    h_harmful_image_mean = hs_harmful_image.mean(axis=0)
    h_harmless_image_mean = hs_harmless_image.mean(axis=0)

    # Calculate refusal vectors
    refusal_text = h_harmful_text_mean - h_harmless_text_mean
    refusal_image = h_harmful_image_mean - h_harmless_image_mean

    num_layers = refusal_text.shape[0]
    norms_text = np.linalg.norm(refusal_text, axis=1)
    norms_image = np.linalg.norm(refusal_image, axis=1)

    # Calculate cosine similarity (layer 0 refusal vectors are zero since every prompt
    # ends in the same token, so its cosine similarity is undefined -> NaN)
    dot_product = np.sum(refusal_text * refusal_image, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine_sim = dot_product / (norms_text * norms_image)

    layers = np.arange(num_layers)

    # Ensure plots directory or save locally
    output_dir = "."

    # Plot norms
    plt.figure()
    plt.plot(layers, norms_text, label='Text Norm', marker='o')
    plt.plot(layers, norms_image, label='Image Norm', marker='x')
    plt.xlabel('Layer')
    plt.ylabel('Norm')
    plt.title('Norm of Refusal Vectors (Text vs Image)')
    plt.legend()
    plt.grid(True)
    torch.save(refusal_text, os.path.join(output_dir, "refusal_text.pt"))
    torch.save(refusal_image, os.path.join(output_dir, "refusal_image.pt"))
    plt.savefig(os.path.join(output_dir, 'norms.png'))
    plt.close()

    # Plot cosine similarity
    plt.figure()
    plt.plot(layers, cosine_sim, label='Cosine Similarity', marker='s', color='purple')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Text and Image Refusal Vectors')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_plot.png'))
    plt.close()

    print(f"Done! Plots saved to {os.path.join(output_dir, 'norms.png')} and {os.path.join(output_dir, 'cosine_similarity_plot.png')}")

if __name__ == "__main__":
    main()
