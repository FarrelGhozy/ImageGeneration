import torch
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)

def load_models_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models to {device}")
    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(device)
    return pipe_txt2img, pipe_inpaint

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory Flushed!")

def set_scheduler(pipe, scheduler_name):
    if scheduler_name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg, num_images=1, scheduler_name="Euler A"):
    pipe = set_scheduler(pipe, scheduler_name)
    results = []
    for i in range(num_images):
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed + i)
        out = pipe(
            prompt=prompt, negative_prompt=neg_prompt,
            generator=generator, guidance_scale=cfg,
            num_inference_steps=steps
        )
        results.append(out.images[0])
    return results

def run_inpainting(pipe, image, mask, prompt, strength):
    if image.mode != "RGB": image = image.convert("RGB")
    if mask.mode != "L": mask = mask.convert("L")
    if image.size != mask.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)
    generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(9)
    result = pipe(
        prompt=prompt, image=image, mask_image=mask,
        generator=generator, num_inference_steps=30,
        guidance_scale=7.5, strength=strength
    ).images[0]
    return result

def prepare_outpainting(image, expand_pixels=128):
    w, h = image.size
    new_w = w + 2 * expand_pixels
    new_h = h + 2 * expand_pixels
    new_w -= new_w % 8
    new_h -= new_h % 8
    bg = image.resize((new_w, new_h), resample=Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=50))
    canvas = bg.copy()
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(image, (paste_x, paste_y))
    mask = Image.new("L", (new_w, new_h), 255)
    inner_box = Image.new("L", (w, h), 0)
    mask.paste(inner_box, (paste_x, paste_y))
    return canvas, mask
