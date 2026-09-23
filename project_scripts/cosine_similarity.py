import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Neutral wrapper for the image+text condition: the request is only in the image,
# the text is the same instruction for every sample.
IMAGE_WRAPPER_TEXT = "Respond to the request in the image."

def load_data(json_path, img_dir, prefix, max_samples=400):
    with open(json_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)[:max_samples]

    images = []
    for i in range(len(prompts)):
        img_path = os.path.join(img_dir, f"{prefix}_{i:03d}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} not found. Regenerate dataset2/ with make_dataset2.py.")
        images.append(Image.open(img_path).convert('RGB'))
    return prompts, images

def get_hidden_states_text(model, processor, prompts, device):
    hidden_states_all = []
    for text in tqdm(prompts, desc="Text processing"):
        prompt_text = f"USER: {text}\nASSISTANT:"
        inputs = processor(text=prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        layer_hs = []
        for hs in outputs.hidden_states:
            layer_hs.append(hs[0, -1, :].cpu().numpy().astype(np.float32))
        hidden_states_all.append(layer_hs)
    return np.array(hidden_states_all)

def get_hidden_states_image(model, processor, images, device):
    hidden_states_all = []
    for img in tqdm(images, desc="Image processing"):
        prompt_text = f"USER: <image>\n{IMAGE_WRAPPER_TEXT}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        layer_hs = []
        for hs in outputs.hidden_states:
            layer_hs.append(hs[0, -1, :].cpu().numpy().astype(np.float32))
        hidden_states_all.append(layer_hs)
    return np.array(hidden_states_all)

def main():
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
    
    num_samples = 400
    
    harmful_json = "dataset2/harmful.json"
    harmful_dir = "dataset2/harmful_images"
    harmful_prompts, harmful_images = load_data(harmful_json, harmful_dir, "harmful", max_samples=num_samples)
    
    harmless_json = "dataset2/harmless.json"
    harmless_dir = "dataset2/harmless"
    harmless_prompts, harmless_images = load_data(harmless_json, harmless_dir, "harmless", max_samples=num_samples)
    
    print("Extracting hidden states for harmful texts...")
    hs_harmful_text = get_hidden_states_text(model, processor, harmful_prompts, device)
    
    print("Extracting hidden states for harmless texts...")
    hs_harmless_text = get_hidden_states_text(model, processor, harmless_prompts, device)
    
    print("Extracting hidden states for harmful images...")
    hs_harmful_image = get_hidden_states_image(model, processor, harmful_images, device)
    
    print("Extracting hidden states for harmless images...")
    hs_harmless_image = get_hidden_states_image(model, processor, harmless_images, device)
    
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
    
    # Calculate cosine similarity
    dot_product = np.sum(refusal_text * refusal_image, axis=1)
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
