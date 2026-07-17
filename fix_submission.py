#!/usr/bin/env python3
"""
Fix script for BFGAI Image Generation submission.
Regenerates all images with correct prompts and implements Two-Stage Generation.
"""
import torch, gc, numpy as np, matplotlib.pyplot as plt, os
from PIL import Image, ImageDraw
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler, DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, PNDMScheduler
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
print(f'Device: {device}, dtype: {dtype}')

PROMPT = 'An astronaut standing on the surface of the Moon, wearing a spacesuit, Earth in the background, stars, detailed, cinematic lighting'
NEG = 'photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration'
SEED = 222

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

def unload(pipe):
    if pipe is not None:
        del pipe
    clear_mem()

def generate_simple_image(pipe, prompt, neg, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe(prompt=prompt, negative_prompt=neg, generator=gen).images[0]

def generate_advanced_image(pipe, prompt, neg, seed, gs, steps):
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=neg,
        generator=gen, guidance_scale=gs,
        num_inference_steps=steps
    ).images[0]

# ============================================================
# KRITERIA 1: Generate images
# ============================================================
print("=" * 60)
print("KRITERIA 1: Text-to-Image")
print("=" * 60)

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype, safety_checker=None
).to(device)
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)
print('Model loaded')

print("Generating simple_image.png...")
img_simple = generate_simple_image(pipe, PROMPT, NEG, SEED)
img_simple.save('simple_image.png')
print("simple_image.png saved")

print("Generating advanced_image.png...")
img_adv = generate_advanced_image(pipe, PROMPT, NEG, SEED, 7.5, 30)
img_adv.save('advanced_image.png')
print("advanced_image.png saved")

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(img_simple); ax[0].set_title('Simple'); ax[0].axis('off')
ax[1].imshow(img_adv); ax[1].set_title('Advanced'); ax[1].axis('off')
plt.tight_layout(); plt.savefig('comparison_simple_advanced.png', dpi=150)
print("Kriteria 1: DONE")

# ============================================================
# KRITERIA 2: Image-to-Image (Inpainting + Two-Stage)
# ============================================================
print("\n" + "=" * 60)
print("KRITERIA 2: Image-to-Image")
print("=" * 60)

# Unload base pipe
unload(pipe)

# Load inpainting model
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    torch_dtype=dtype, safety_checker=None
).to(device)
pipe_inpaint.enable_attention_slicing()
pipe_inpaint.set_progress_bar_config(disable=True)
print('Inpainting model loaded')

def inpaint_engine(pipe, img, mask, prompt, seed=9, steps=30, gs=7.5):
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt=prompt, image=img, mask_image=mask,
        generator=gen, num_inference_steps=steps,
        guidance_scale=gs
    ).images[0]

# Use advanced_image.png (astronaut) as base for inpainting
print("Loading advanced_image.png for inpainting...")
base = Image.open('advanced_image.png').convert('RGB').resize((512, 512))
base.save('base_inpaint.png')

# Hardcoded mask - area langit kanan atas
mask = Image.new('RGB', (512, 512), (0,0,0))
ImageDraw.Draw(mask).ellipse([(350,60),(460,200)], fill=(255,255,255))
mask.save('manual_mask.png')

print("Running inpainting...")
res = inpaint_engine(
    pipe_inpaint, base, mask.convert('L'),
    'A broken satellite drifting in space, damaged solar panels, detailed, cinematic',
    seed=9
)
res.save('inpainted_satellite.png')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(base); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(mask); axes[1].set_title('Mask'); axes[1].axis('off')
axes[2].imshow(res); axes[2].set_title('Inpainted'); axes[2].axis('off')
plt.tight_layout(); plt.savefig('inpainting_result.png', dpi=150)
print("Inpainting done")

# ============================================================
# Two-Stage Generation
# ============================================================
print("\n" + "=" * 60)
print("Two-Stage Generation (Base + Refiner)")
print("=" * 60)

unload(pipe_inpaint); pipe_inpaint = None
clear_mem()

pipe_base = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype, safety_checker=None
).to(device)
pipe_base.enable_attention_slicing()
pipe_base.set_progress_bar_config(disable=True)

pipe_refiner = StableDiffusionImg2ImgPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=dtype, safety_checker=None
).to(device)
pipe_refiner.enable_attention_slicing()
pipe_refiner.set_progress_bar_config(disable=True)
print('Base + Refiner pipelines loaded for two-stage')

clear_mem()

# Stage 1: Base pipeline with denoising_end=0.8, output latent
print("Stage 1: Generating latent with denoising_end=0.8...")
gen = torch.Generator(device=device).manual_seed(222)
out_s1 = pipe_base(
    prompt=PROMPT, negative_prompt=NEG,
    generator=gen, guidance_scale=7.5,
    num_inference_steps=50, denoising_end=0.8,
    output_type='latent'
)
latents = out_s1.images

# Decode latent ke PIL image
with torch.no_grad():
    scaled = latents / pipe_base.vae.config.scaling_factor
    img_t = pipe_base.vae.decode(scaled, return_dict=False)[0]
    img_t = ((img_t / 2 + 0.5).clamp(0, 1))
img_stage1 = Image.fromarray(
    (img_t[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
)
img_stage1.save('stage1_result.png')

del latents, out_s1, scaled, img_t; clear_mem()

# Stage 2: Img2Img refinement with denoising_start=0.8
print("Stage 2: Refining with denoising_start=0.8...")
gen2 = torch.Generator(device=device).manual_seed(222)
refined = pipe_refiner(
    prompt=PROMPT, negative_prompt=NEG,
    image=img_stage1, generator=gen2,
    guidance_scale=7.5, num_inference_steps=50,
    denoising_start=0.8
).images[0]
refined.save('twostage.png')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img_stage1); axes[0].set_title('Stage 1 (denoising_end=0.8)'); axes[0].axis('off')
axes[1].imshow(img_adv); axes[1].set_title('Standard T2I'); axes[1].axis('off')
axes[2].imshow(refined); axes[2].set_title('Stage 2 Refined (denoising_start=0.8)'); axes[2].axis('off')
plt.tight_layout(); plt.savefig('twostage_compare.png', dpi=150)
print("Two-Stage Generation: DONE")

# Cleanup
unload(pipe_base); unload(pipe_refiner)
print("All done!")
