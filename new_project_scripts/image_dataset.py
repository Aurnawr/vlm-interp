import os
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont

def create_image_dataset():
    input_file = "dataset2/harmless.json"
    output_dir = "harmless"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the harmless prompts
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    # Try to load a TrueType font, otherwise fallback to the default bitmap font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    width, height = 800, 600
    
    for i, prompt in enumerate(prompts):
        # Create a blank white image
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        
        # Wrap text to fit the image reasonably
        wrapped_text = textwrap.fill(prompt, width=70)
        
        # Draw the text starting at x=20, y=20
        draw.text((20, 20), wrapped_text, fill="black", font=font)
        
        output_path = os.path.join(output_dir, f"harmless_{i:03d}.png")
        img.save(output_path)

    print(f"Successfully generated {len(prompts)} images in '{output_dir}'.")

if __name__ == "__main__":
    create_image_dataset()
