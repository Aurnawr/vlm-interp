import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Neutral wrapper for the image+text condition: the request is only in the image,
# the text is the same instruction for every sample.
IMAGE_WRAPPER_TEXT = "Respond to the request in the image."

def load_data(json_path, img_dir, prefix, max_samples=50):
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
        # Text-only format for LLaVA
        prompt_text = f"USER: {text}\nASSISTANT:"
        inputs = processor(text=prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True) # outputs all the hidden states 
            
        layer_hs = []
        for hs in outputs.hidden_states:
            layer_hs.append(hs[0, -1, :].cpu().numpy().astype(np.float32))
        hidden_states_all.append(layer_hs)
        
    return np.array(hidden_states_all)

def get_hidden_states_image(model, processor, images, device):
    hidden_states_all = []
    
    for img in tqdm(images, desc="Image processing"):
        # Image + neutral text format for LLaVA
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
    # Load with float16 to save memory, auto device_map if bitsandbytes/accelerate is installed
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    num_samples = 400 # Limit to 40 samples per class to keep it fast/avoid OOM for demonstration
    print(f"Loading subset of data ({num_samples} samples per class)...")
    
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
    
    num_layers = hs_harmful_text.shape[1]
    
    output_dir = "pca_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 6 evenly spaced layers from the embedding layer to the final layer
    layers_to_plot = np.linspace(0, num_layers - 1, 6).round().astype(int).tolist()
    
    print(f"Generating PCA plots for layers: {layers_to_plot}")
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
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 7))
        plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Harmful Text', alpha=0.7, color='red', marker='o')
        plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Harmless Text', alpha=0.7, color='green', marker='o')
        plt.scatter(X_pca[y==2, 0], X_pca[y==2, 1], label='Harmful Image', alpha=0.7, color='darkred', marker='x')
        plt.scatter(X_pca[y==3, 0], X_pca[y==3, 1], label='Harmless Image', alpha=0.7, color='darkgreen', marker='x')
        
        plt.title(f"LLaVA-7B Hidden States PCA (Layer {layer_idx})")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"pca_layer_{layer_idx}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

if __name__ == "__main__":
    main()
