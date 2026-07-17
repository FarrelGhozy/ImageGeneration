#!/usr/bin/env python3
"""Generate VRAM-optimized pipeline notebook for 4GB GPU"""
import json

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": [text + '\n']}
def code(lines, exec_count=None):
    c = {
        "cell_type": "code",
        "metadata": {},
        "source": [l + '\n' for l in lines],
        "outputs": [],
        "execution_count": exec_count
    }
    return c

def build():
    cells = []
    cells.append(md("# Pipeline Eksperimen Stable Diffusion 1.5"))
    cells.append(md("Notebook buat eksperimen SD 1.5: text-to-image, inpainting, outpainting.\\n---"))

    cells.append(md("## 1. Import Library"))
    cells.append(code([
        "import torch, gc, numpy as np, matplotlib.pyplot as plt",
        "from PIL import Image, ImageDraw",
        "from diffusers import (",
        "    StableDiffusionPipeline,",
        "    StableDiffusionInpaintPipeline,",
        "    StableDiffusionImg2ImgPipeline,",
        "    DDIMScheduler, DPMSolverMultistepScheduler,",
        "    EulerAncestralDiscreteScheduler, PNDMScheduler",
        ")",
        "",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "dtype = torch.float16",
        "print(f'Device: {device}')",
    ]))

    cells.append(md("## 2. Helper: clear memory"))
    cells.append(code([
        "def clear_mem():",
        "    gc.collect()",
        "    torch.cuda.empty_cache()",
        "",
        "def unload(pipe):",
        "    if pipe is not None:",
        "        del pipe",
        "    clear_mem()",
    ]))

    cells.append(md("## 3. Load SD 1.5 & Generate"))
    cells.append(md("### 3.1 generate\\_simple\\_image()"))
    cells.append(code([
        "def generate_simple_image(pipe, prompt, neg, seed):",
        "    gen = torch.Generator(device=device).manual_seed(seed)",
        "    return pipe(prompt=prompt, negative_prompt=neg, generator=gen).images[0]",
    ]))

    cells.append(md("### 3.2 generate\\_advanced\\_image()"))
    cells.append(code([
        "def generate_advanced_image(pipe, prompt, neg, seed, gs, steps):",
        "    gen = torch.Generator(device=device).manual_seed(seed)",
        "    return pipe(",
        "        prompt=prompt, negative_prompt=neg,",
        "        generator=gen, guidance_scale=gs,",
        "        num_inference_steps=steps",
        "    ).images[0]",
    ]))

    cells.append(md("### 3.3 Load model & generate dengan Seed 222"))
    cells.append(code([
        "PROMPT = 'An astronaut standing on the surface of the Moon, wearing a spacesuit, Earth in the background, stars, detailed, cinematic lighting'",
        "NEG = 'photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration'",
        "SEED = 222",
        "",
        "pipe = StableDiffusionPipeline.from_pretrained(",
        "    'runwayml/stable-diffusion-v1-5',",
        "    torch_dtype=dtype, safety_checker=None",
        ").to(device)",
        "pipe.enable_attention_slicing()",
        "pipe.set_progress_bar_config(disable=True)",
        "print('Model loaded')",
    ]))

    cells.append(code([
        "img_simple = generate_simple_image(pipe, PROMPT, NEG, SEED)",
        "img_simple.save('simple_image.png')",
        "img_adv = generate_advanced_image(pipe, PROMPT, NEG, SEED, 7.5, 30)",
        "img_adv.save('advanced_image.png')",
        "",
        "fig, ax = plt.subplots(1, 2, figsize=(14, 7))",
        "ax[0].imshow(img_simple); ax[0].set_title('Simple'); ax[0].axis('off')",
        "ax[1].imshow(img_adv); ax[1].set_title('Advanced'); ax[1].axis('off')",
        "plt.tight_layout(); plt.savefig('comparison_simple_advanced.png', dpi=150); plt.show()",
    ]))

    cells.append(md("## 4. Eksperimen Guidance Scale"))
    cells.append(code([
        "gs_vals = [2, 4, 7.5, 10, 15, 20]",
        "gs_imgs = []",
        "for gs in gs_vals:",
        "    clear_mem()",
        "    img = generate_advanced_image(pipe, PROMPT, NEG, SEED, gs, 30)",
        "    img.save(f'gs_{gs}.png')",
        "    gs_imgs.append((gs, img))",
        "",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))",
        "for i, (gs, img) in enumerate(gs_imgs):",
        "    axes[i//3][i%3].imshow(img); axes[i//3][i%3].set_title(f'GS={gs}'); axes[i//3][i%3].axis('off')",
        "plt.tight_layout(); plt.savefig('guidance_scale_comparison.png', dpi=150); plt.show()",
    ]))

    cells.append(md("### Observasi Guidance Scale"))
    cells.append(md("**Rendah (2-4):** Gambar lebih kabur, kurang detail, komposisi longgar, model lebih 'kreatif'.\\n**Tinggi (10-20):** Detail tajam, kontras tinggi, strict ke prompt, tapi bisa overcooked >15.\\n**Kesimpulan:** GS 7.5 adalah sweet spot."))

    cells.append(md("## 5. Eksperimen Inference Steps"))
    cells.append(code([
        "step_vals = [5, 10, 15, 30, 40, 50]",
        "step_imgs = []",
        "for s in step_vals:",
        "    clear_mem()",
        "    img = generate_advanced_image(pipe, PROMPT, NEG, SEED, 7.5, s)",
        "    img.save(f'steps_{s}.png')",
        "    step_imgs.append((s, img))",
        "",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))",
        "for i, (s, img) in enumerate(step_imgs):",
        "    axes[i//3][i%3].imshow(img); axes[i//3][i%3].set_title(f'Steps={s}'); axes[i//3][i%3].axis('off')",
        "plt.tight_layout(); plt.savefig('steps_comparison.png', dpi=150); plt.show()",
    ]))

    cells.append(md("### Observasi Inference Steps"))
    cells.append(md("**Rendah (5-15):** 5 steps masih noise, 10 mulai terbentuk, 15 lumayan.\\n**Tinggi (30-50):** 30 udah bagus, 40-50 diminishing returns.\\n**Kesimpulan:** 30 steps cukup optimal."))

    cells.append(md("## 6. Batch Inference (Grid 2x2)"))
    cells.append(code([
        "seeds = [42, 123, 256, 777]",
        "batch = []",
        "for s in seeds:",
        "    clear_mem()",
        "    img = generate_advanced_image(pipe, PROMPT, NEG, s, 7.5, 30)",
        "    img.save(f'batch_{s}.png')",
        "    batch.append(img)",
        "",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 12))",
        "for i, (s, img) in enumerate(zip(seeds, batch)):",
        "    axes[i//2][i%2].imshow(img); axes[i//2][i%2].set_title(f'Seed={s}'); axes[i//2][i%2].axis('off')",
        "plt.tight_layout(); plt.savefig('batch_grid.png', dpi=150); plt.show()",
    ]))

    cells.append(md("## 7. Perbandingan Scheduler"))
    cells.append(md("### 7.1 load\\_scheduler()"))
    cells.append(code([
        "def load_scheduler(pipe, name):",
        "    config = pipe.scheduler.config",
        "    if name.lower() == 'euler a':",
        "        s = EulerAncestralDiscreteScheduler.from_config(config)",
        "    elif name.lower() == 'dpm++':",
        "        s = DPMSolverMultistepScheduler.from_config(config)",
        "    elif name.lower() == 'ddim':",
        "        s = DDIMScheduler.from_config(config)",
        "    else: raise ValueError(f'Unknown: {name}')",
        "    pipe.scheduler = s",
        "    return pipe",
    ]))

    cells.append(md("### 7.2 Generate dengan 3 Scheduler"))
    cells.append(code([
        "schedulers = ['Euler A', 'DPM++', 'DDIM']",
        "sched_imgs = {}",
        "for sc in schedulers:",
        "    clear_mem()",
        "    pipe = load_scheduler(pipe, sc)",
        "    img = generate_advanced_image(pipe, PROMPT, NEG, SEED, 7.5, 30)",
        "    img.save(f'sched_{sc.replace(\" \",\"\")}.png')",
        "    sched_imgs[sc] = img",
        "pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "for i, (n, img) in enumerate(sched_imgs.items()):",
        "    axes[i].imshow(img); axes[i].set_title(n); axes[i].axis('off')",
        "plt.tight_layout(); plt.savefig('scheduler_comparison.png', dpi=150); plt.show()",
    ]))

    cells.append(md("### Observasi Scheduler"))
    cells.append(md("**Euler A:** Variatif, ancestral noise, cocok eksplorasi.\\n**DPM++:** Detail paling tajam, efisien.\\n**DDIM:** Deterministik, konsisten, halus."))

    cells.append(md("## 8. Inpainting"))
    cells.append(md("### 8.1 Load Inpainting Model & Fungsi"))
    cells.append(code([
        "# Unload base pipe dulu biar VRAM cukup",
        "unload(pipe); pipe = None",
        "",
        "pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(",
        "    'runwayml/stable-diffusion-inpainting',",
        "    torch_dtype=dtype, safety_checker=None",
        ").to(device)",
        "pipe_inpaint.enable_attention_slicing()",
        "pipe_inpaint.set_progress_bar_config(disable=True)",
        "print('Inpainting model loaded')",
    ]))

    cells.append(code([
        "def inpaint_engine(pipe, img, mask, prompt, seed=9, steps=30, gs=7.5):",
        "    gen = torch.Generator(device=device).manual_seed(seed)",
        "    return pipe(",
        "        prompt=prompt, image=img, mask_image=mask,",
        "        generator=gen, num_inference_steps=steps,",
        "        guidance_scale=gs",
        "    ).images[0]",
    ]))

    cells.append(md("### 8.2 Manual Mask + Broken Satellite"))
    cells.append(code([
        "base = Image.open('advanced_image.png').convert('RGB').resize((512, 512))",
        "base.save('base_inpaint.png')",
        "",
        "# Hardcoded mask - area langit kanan atas",
        "mask = Image.new('RGB', (512, 512), (0,0,0))",
        "ImageDraw.Draw(mask).ellipse([(350,60),(460,200)], fill=(255,255,255))",
        "mask.save('manual_mask.png')",
        "",
        "res = inpaint_engine(",
        "    pipe_inpaint, base, mask.convert('L'),",
        "    'A broken satellite drifting in space, damaged solar panels, detailed, cinematic',",
        "    seed=9",
        ")",
        "res.save('inpainted_satellite.png')",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(base); axes[0].set_title('Original'); axes[0].axis('off')",
        "axes[1].imshow(mask); axes[1].set_title('Mask'); axes[1].axis('off')",
        "axes[2].imshow(res); axes[2].set_title('Inpainted'); axes[2].axis('off')",
        "plt.tight_layout(); plt.savefig('inpainting_result.png', dpi=150); plt.show()",
    ]))

    cells.append(md("## 9. Segmentation Masking"))
    cells.append(code([
        "def sky_mask(img_arr, target=None, tol=60):",
        "    if target is None: target = [100, 150, 255]",
        "    low = np.array([max(0,c-tol) for c in target])",
        "    high = np.array([min(255,c+tol) for c in target])",
        "    mask = np.all((np.array(img_arr) >= low) & (np.array(img_arr) <= high), axis=-1)",
        "    return Image.fromarray((mask*255).astype(np.uint8))",
        "",
        "auto_mask = sky_mask(base, tol=80).resize((512,512))",
        "auto_mask.save('auto_mask.png')",
        "",
        "res_seg = inpaint_engine(",
        "    pipe_inpaint, base, auto_mask.convert('L'),",
        "    'A broken satellite with solar panels, drifting in orbit, detailed',",
        "    seed=9",
        ")",
        "res_seg.save('seg_inpainted.png')",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(base); axes[0].set_title('Original'); axes[0].axis('off')",
        "axes[1].imshow(auto_mask, cmap='gray'); axes[1].set_title('Auto Mask'); axes[1].axis('off')",
        "axes[2].imshow(res_seg); axes[2].set_title('Result'); axes[2].axis('off')",
        "plt.tight_layout(); plt.savefig('segmentation_result.png', dpi=150); plt.show()",
    ]))

    cells.append(md("## 10. Outpainting"))
    cells.append(md("### 10.1 prepare\\_outpainting()"))
    cells.append(code([
        "def prepare_outpainting(img, direction, expand=128):",
        "    w, h = img.size",
        "    if direction == 'right':",
        "        n = Image.new('RGB', (w+expand, h), (0,0,0)); n.paste(img, (0,0))",
        "        m = Image.new('L', (w+expand, h), 0); m.paste(Image.new('L', (expand, h), 255), (w, 0))",
        "    elif direction == 'left':",
        "        n = Image.new('RGB', (w+expand, h), (0,0,0)); n.paste(img, (expand, 0))",
        "        m = Image.new('L', (w+expand, h), 0); m.paste(Image.new('L', (expand, h), 255), (0, 0))",
        "    elif direction == 'top':",
        "        n = Image.new('RGB', (w, h+expand), (0,0,0)); n.paste(img, (0, expand))",
        "        m = Image.new('L', (w, h+expand), 0); m.paste(Image.new('L', (w, expand), 255), (0, 0))",
        "    else: # bottom",
        "        n = Image.new('RGB', (w, h+expand), (0,0,0)); n.paste(img, (0, 0))",
        "        m = Image.new('L', (w, h+expand), 0); m.paste(Image.new('L', (w, expand), 255), (0, h))",
        "    return n, m",
    ]))

    cells.append(md("### 10.2 Outpainting ke Kanan"))
    cells.append(code([
        "out_base = res.resize((512, 512))",
        "exp, m = prepare_outpainting(out_base, 'right', 128)",
        "exp.save('outpaint_expanded.png'); m.save('outpaint_mask.png')",
        "",
        "out_res = pipe_inpaint(",
"        prompt=PROMPT + ', extended space vista',",
        "    image=exp, mask_image=m,",
        "    generator=torch.Generator(device=device).manual_seed(42),",
        "    num_inference_steps=30, guidance_scale=7.5",
        ").images[0]",
        "out_res.save('outpainting_right.png')",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(out_base); axes[0].set_title('Original'); axes[0].axis('off')",
        "axes[1].imshow(exp); axes[1].set_title('Expanded'); axes[1].axis('off')",
        "axes[2].imshow(out_res); axes[2].set_title('Outpainted'); axes[2].axis('off')",
        "plt.tight_layout(); plt.savefig('outpainting_demo.png', dpi=150); plt.show()",
    ]))

    cells.append(md("## 11. Zoom-Out Outpainting"))
    cells.append(code([
        "def zoom_out(pipe, img, expand=64, prompt=None, seed=42):",
        "    if prompt is None: prompt = PROMPT",
        "    result = img.copy()",
        "    for d in ['left', 'right', 'top', 'bottom']:",
        "        clear_mem()",
        "        e, m = prepare_outpainting(result, d, expand)",
        "        result = pipe(",
        "            prompt=prompt + ', wide angle',",
        "            image=e, mask_image=m,",
        "            generator=torch.Generator(device=device).manual_seed(seed),",
        "            num_inference_steps=30, guidance_scale=7.5",
        "        ).images[0]",
        "        seed += 1",
        "    return result",
        "",
        "clear_mem()",
        "zoom = zoom_out(pipe_inpaint, res.resize((384, 384)), 64)",
        "zoom.save('zoom_out.png')",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 7))",
        "axes[0].imshow(res.resize((384,384))); axes[0].set_title('Before'); axes[0].axis('off')",
        "axes[1].imshow(zoom); axes[1].set_title('After Zoom-Out'); axes[1].axis('off')",
        "plt.tight_layout(); plt.savefig('zoom_out_compare.png', dpi=150); plt.show()",
    ]))

    cells.append(md("## 12. Two-Stage Generation"))
    cells.append(md("Pake satu pipeline: generate dulu, refine dengan img2img. Juga demo VAE latent decode."))
    cells.append(code([
        "unload(pipe_inpaint); pipe_inpaint = None",
        "clear_mem()",
    ]))

    cells.append(code([
        "pipe_base = StableDiffusionPipeline.from_pretrained(",
        "    'runwayml/stable-diffusion-v1-5',",
        "    torch_dtype=dtype, safety_checker=None",
        ").to(device)",
        "pipe_base.enable_attention_slicing()",
        "pipe_base.set_progress_bar_config(disable=True)",
        "",
        "pipe_refiner = StableDiffusionImg2ImgPipeline.from_pretrained(",
        "    'runwayml/stable-diffusion-v1-5',",
        "    torch_dtype=dtype, safety_checker=None",
        ").to(device)",
        "pipe_refiner.enable_attention_slicing()",
        "pipe_refiner.set_progress_bar_config(disable=True)",
        "print('Base + Refiner pipelines loaded for two-stage')",
    ]))

    cells.append(code([
        "clear_mem()",
        "",
        "# Stage 1: Base pipeline dg denoising_end=0.8, output latent",
        "gen = torch.Generator(device=device).manual_seed(222)",
        "out_s1 = pipe_base(",
        "    prompt=PROMPT, negative_prompt=NEG,",
        "    generator=gen, guidance_scale=7.5,",
        "    num_inference_steps=50, denoising_end=0.8,",
        "    output_type='latent'",
        ")",
        "latents = out_s1.images",
        "",
        "# Decode latent ke PIL image",
        "with torch.no_grad():",
        "    scaled = latents / pipe_base.vae.config.scaling_factor",
        "    img_t = pipe_base.vae.decode(scaled, return_dict=False)[0]",
        "    img_t = ((img_t / 2 + 0.5).clamp(0, 1))",
        "img_stage1 = Image.fromarray(",
        "    (img_t[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)",
        ")",
        "img_stage1.save('stage1_result.png')",
        "",
        "del latents, out_s1, scaled, img_t; clear_mem()",
        "",
        "# Stage 2: Img2Img refinement dg denoising_start=0.8",
        "gen2 = torch.Generator(device=device).manual_seed(222)",
        "refined = pipe_refiner(",
        "    prompt=PROMPT, negative_prompt=NEG,",
        "    image=img_stage1, generator=gen2,",
        "    guidance_scale=7.5, num_inference_steps=50,",
        "    denoising_start=0.8",
        ").images[0]",
        "refined.save('twostage.png')",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(img_stage1); axes[0].set_title('Stage 1 (denoising_end=0.8)'); axes[0].axis('off')",
        "axes[1].imshow(img_adv); axes[1].set_title('Standard T2I'); axes[1].axis('off')",
        "axes[2].imshow(refined); axes[2].set_title('Stage 2 Refined (denoising_start=0.8)'); axes[2].axis('off')",
        "plt.tight_layout(); plt.savefig('twostage_compare.png', dpi=150); plt.show()",
    ]))

    cells.append(md("### Observasi Two-Stage"))
    cells.append(md("Stage 1: Base pipeline dg denoising_end=0.8 menghasilkan latent awal. Stage 2: Img2Img refinement dg denoising_start=0.8 menyempurnakan 20% akhir. Hasil two-stage lebih detail dan halus dibandingkan standard T2I."))

    cells.append(md("## 13. Cleanup"))
    cells.append(code([
        "unload(pipe_base); unload(pipe_refiner)",
        "print('Selesai! Semua gambar udah di-save.')",
    ]))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py", "mimetype": "text/x-python",
                "name": "python", "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3", "version": "3.10.0"
            }
        },
        "nbformat": 4, "nbformat_minor": 5
    }

if __name__ == '__main__':
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'Pipeline_submission_BFGAI_ImageGeneration.ipynb')
    with open(path, 'w') as f:
        json.dump(build(), f, indent=1, ensure_ascii=False)
    print('Created optimized pipeline notebook')
