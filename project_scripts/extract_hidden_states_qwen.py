import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pca_plots_llava import load_data, IMAGE_WRAPPER_TEXT

# Qwen2.5-VL-7B-Instruct counterpart of the LLaVA extraction in tsne_plots_llava.py.
# Same dataset, same 4 conditions, same last-token readout at every layer, so the
# cached npz has the same layout: [num_samples, num_layers, hidden_dim] per condition
# (29 layers incl. embeddings, hidden_dim 3584).

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_PATH = os.path.join(REPO_ROOT, "qwen-results", "hidden_states_qwen.npz")


def last_token_states(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [hs[0, -1, :].float().cpu().numpy() for hs in outputs.hidden_states]


# Prompts go through the Qwen chat template (which adds the default system prompt) and
# end in "<|im_start|>assistant\n", so the last token is identical for every sample and
# the layer-0 refusal vector is zero, as with LLaVA.

def get_hidden_states_text(model, processor, prompts, device):
    hidden_states_all = []
    for text in tqdm(prompts, desc="Text processing"):
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], return_tensors="pt").to(device)
        hidden_states_all.append(last_token_states(model, inputs))
    return np.array(hidden_states_all, dtype=np.float32)


def get_hidden_states_image(model, processor, images, device):
    hidden_states_all = []
    for img in tqdm(images, desc="Image processing"):
        # Image + neutral text: the request is only in the image
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": IMAGE_WRAPPER_TEXT},
        ]}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[img], return_tensors="pt").to(device)
        hidden_states_all.append(last_token_states(model, inputs))
    return np.array(hidden_states_all, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--gpu_mem", default="13GiB",
                        help="GPU budget; the rest of the ~16.6GB bf16 weights is offloaded to CPU")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # bf16, not fp16: Qwen2.5 overflows in fp16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_mem, "cpu": "96GiB"} if device == "cuda" else None,
    )
    model.eval()

    data_dir = os.path.join(REPO_ROOT, "dataset2")
    harmful_prompts, harmful_images = load_data(
        os.path.join(data_dir, "harmful.json"), os.path.join(data_dir, "harmful_images"),
        "harmful", max_samples=args.num_samples)
    harmless_prompts, harmless_images = load_data(
        os.path.join(data_dir, "harmless.json"), os.path.join(data_dir, "harmless"),
        "harmless", max_samples=args.num_samples)

    print("Extracting hidden states for harmful texts...")
    hs_harmful_text = get_hidden_states_text(model, processor, harmful_prompts, device)
    print("Extracting hidden states for harmless texts...")
    hs_harmless_text = get_hidden_states_text(model, processor, harmless_prompts, device)
    print("Extracting hidden states for harmful images...")
    hs_harmful_image = get_hidden_states_image(model, processor, harmful_images, device)
    print("Extracting hidden states for harmless images...")
    hs_harmless_image = get_hidden_states_image(model, processor, harmless_images, device)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH,
             harmful_text=hs_harmful_text, harmless_text=hs_harmless_text,
             harmful_image=hs_harmful_image, harmless_image=hs_harmless_image)
    print(f"Saved hidden states {hs_harmful_text.shape} per condition to {CACHE_PATH}")


if __name__ == "__main__":
    main()
