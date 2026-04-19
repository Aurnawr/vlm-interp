import argparse
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description="Calculate cosine similarity of a prompt with refusal vectors.")
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt to feed into the model.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path to include in the prompt.")
    parser.add_argument("--refusal-text", type=str, default="refusal_text.pt", help="Path to text refusal vector.")
    parser.add_argument("--refusal-image", type=str, default="refusal_image.pt", help="Path to image refusal vector.")
    parser.add_argument("--output", type=str, default="interactive_cosine_sim.png", help="Output plot filename.")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load refusal vectors
    try:
        refusal_text = torch.load(args.refusal_text, weights_only=False)
        refusal_image = torch.load(args.refusal_image, weights_only=False)
    except Exception as e:
        print(f"Error loading refusal vectors: {e}")
        print("Make sure you run cosine_similarity.py first so that the .pt files exist.")
        return
        
    num_layers = refusal_text.shape[0]
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    print("\nProcessing user prompt...")
    # Properly evaluate the string to handle \n literals
    try:
        decoded_prompt = ast.literal_eval(f'"{args.prompt}"')
    except Exception:
        decoded_prompt = args.prompt
    
    if args.image:
        print(f"Image: {args.image}")
        img = Image.open(args.image).convert("RGB")
        prompt_text = f"USER: <image>\n{decoded_prompt}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=img, return_tensors="pt").to(device)
    else:
        prompt_text = f"USER: {decoded_prompt}\nASSISTANT:"
        inputs = processor(text=prompt_text, return_tensors="pt").to(device)
        
    print(f"Prompt sent to model:\n{prompt_text}")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    hidden_states = []
    for hs in outputs.hidden_states:
        # Get the representation of the last token
        layer_hs = hs[0, -1, :].cpu().numpy().astype(np.float32)
        hidden_states.append(layer_hs)
        
    hidden_states = np.array(hidden_states) # [num_layers, hidden_size]
    
    # Normalize and Calculate Cosine Similarities
    norms_rt = np.linalg.norm(refusal_text, axis=1)
    norms_ri = np.linalg.norm(refusal_image, axis=1)
    norms_hs = np.linalg.norm(hidden_states, axis=1)
    
    sim_text = np.sum(hidden_states * refusal_text, axis=1) / (norms_hs * norms_rt)
    sim_image = np.sum(hidden_states * refusal_image, axis=1) / (norms_hs * norms_ri)
    
    layers = np.arange(num_layers)
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, sim_text, label='Similarity w/ Text Refusal', marker='o', color='blue', alpha=0.7)
    plt.plot(layers, sim_image, label='Similarity w/ Image Refusal', marker='x', color='red', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of User Prompt vs Refusal Vectors')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    
    print(f"\nDone! Plot saved to {args.output}")

if __name__ == "__main__":
    main()
