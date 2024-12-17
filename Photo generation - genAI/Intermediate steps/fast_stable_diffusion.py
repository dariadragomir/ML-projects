from diffusers import DiffusionPipeline
import torch

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

prompt = "A very pretty woman"
n_steps_base = 20    # reduced steps for faster processing
n_steps_refiner = 5  # fewer steps for the refiner to save time
high_noise_frac = 0.7
moderate_res = 384   # lower resolution for faster processing

image = base(
    prompt=prompt,
    num_inference_steps=n_steps_base,
    denoising_end=high_noise_frac,
    output_type="latent",
    height=moderate_res,
    width=moderate_res,
).images

# fewer steps to maintain detail without long processing
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps_refiner,
    denoising_start=high_noise_frac,
    image=image,
    height=moderate_res,
    width=moderate_res,
).images[0]

image.save("optimized_final_image.png")
