"""
Image Generation from Text Prompts using Stable Diffusion
References:
1. https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion
2. https://colab.research.google.com/github/robgon-art/e-dall-e/blob/main/DALL_E_Mini_Image_Generator.ipynb
3. https://towardsdatascience.com/e-dall-e-creating-digital-art-with-varying-aspect-ratios-5de260f4713d/
4. https://github.com/faizonly5953/Diffusion-Colab
"""

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def generate_image(prompt, output_path="generated_image.png"):
    # Load the Stable Diffusion pipeline from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    # Generate image
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    user_prompt = input("Enter your text prompt: ")
    generate_image(user_prompt)
