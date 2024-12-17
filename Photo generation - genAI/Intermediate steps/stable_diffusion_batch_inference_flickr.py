import pandas as pd
from diffusers import DiffusionPipeline
import torch

flickr_data = pd.read_csv("flickr.csv", delimiter="|", header=None, names=["image_name", "comment_number", "comment"])

prompts = flickr_data["comment"].head(35).tolist()
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
base.to("mps")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("mps")

n_steps = 20
high_noise_frac = 0.8

for i, prompt in enumerate(prompts):
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    image.save(f"generated_image_{i+1}.png")
