#!/usr/bin/env python3
"""
Generate human-style notebooks for Dicoding submission.
"""

import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text + '\n']}

def code(src_lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + '\n' for l in src_lines],
        "outputs": [],
        "execution_count": None
    }

# ============================================================
# PIPELINE NOTEBOOK
# ============================================================
def build_pipeline():
    cells = []

    cells.append(md("# Pipeline Eksperimen Stable Diffusion 1.5"))
    cells.append(md("Notebook ini gw buat untuk eksperimen pipeline Stable Diffusion 1.5. "
                     "Mulai dari text-to-image, inpainting, sampe outpainting. "
                     "Semua pake model runwayml/stable-diffusion-v1-5 dan varian inpaintingnya.\\n"
                     "\\n"
                     "---"))

    # --- Install ---
    cells.append(md("## 1. Install dulu dependensinya"))
    cells.append(code([
        "!pip install diffusers transformers accelerate torch pillow matplotlib numpy opencv-python scipy",
    ]))

    # --- Import ---
    cells.append(md("## 2. Import library"))
    cells.append(code([
        "import torch",
        "import numpy as np",
        "from diffusers import (",
        "    StableDiffusionPipeline,",
        "    StableDiffusionInpaintPipeline,",
        "    StableDiffusionImg2ImgPipeline,",
        "    DDIMScheduler,",
        "    DPMSolverMultistepScheduler,",
        "    EulerAncestralDiscreteScheduler,",
        "    PNDMScheduler,",
        ")",
        "from PIL import Image, ImageDraw",
        "import matplotlib.pyplot as plt",
        "import gc",
        "import os",
        "",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "print(f'Pake device: {device}')",
    ]))

    # --- Load SD 1.5 ---
    cells.append(md("## 3. Load Stable Diffusion 1.5"))
    cells.append(md("Model utama yang bakal dipake buat generate gambar dari teks."))
    cells.append(code([
        "MODEL_ID = 'runwayml/stable-diffusion-v1-5'",
        "",
        "pipe = StableDiffusionPipeline.from_pretrained(",
        "    MODEL_ID,",
        "    torch_dtype=torch.float16,",
        "    safety_checker=None,",
        "    requires_safety_checker=False,",
        ")",
        "pipe = pipe.to(device)",
        "pipe.set_progress_bar_config(disable=True)",
        "print('Model siap!')",
    ]))

    # --- generate_simple_image ---
    cells.append(md("## 4. Fungsi Text-to-Image"))
    cells.append(md("### 4.1 generate\\_simple\\_image()"))
    cells.append(md("Fungsi dasar, cuma pake 3 parameter: prompt, negative\\_prompt, sama seed."))
    cells.append(code([
        "def generate_simple_image(pipe, prompt, negative_prompt, seed):",
        "    generator = torch.Generator(device=device).manual_seed(seed)",
        "    result = pipe(",
        "        prompt=prompt,",
        "        negative_prompt=negative_prompt,",
        "        generator=generator,",
        "    )",
        "    return result.images[0]",
    ]))

    # --- generate_advanced_image ---
    cells.append(md("### 4.2 generate\\_advanced\\_image()"))
    cells.append(md("Yang ini pake parameter tambahan: guidance\\_scale sama num\\_inference\\_steps."))
    cells.append(code([
        "def generate_advanced_image(pipe, prompt, negative_prompt, seed, guidance_scale, num_inference_steps):",
        "    generator = torch.Generator(device=device).manual_seed(seed)",
        "    result = pipe(",
        "        prompt=prompt,",
        "        negative_prompt=negative_prompt,",
        "        generator=generator,",
        "        guidance_scale=guidance_scale,",
        "        num_inference_steps=num_inference_steps,",
        "    )",
        "    return result.images[0]",
    ]))

    # --- Generate with seed 222 ---
    cells.append(md("### 4.3 Generate pake Seed 222"))
    cells.append(md("Promptnya sama biar bisa dibandingin secara objektif.\\n"
                     "\\n"
                     "**Prompt:** \\\"An astronaut standing on the surface of the Moon, wearing a spacesuit, Earth in the background, stars, detailed, cinematic lighting\\\"\\n"
                     "**Negative Prompt:** sesuai ketentuan tugas\\n"
                     "**Seed:** 222"))
    cells.append(code([
        "PROMPT = 'An astronaut standing on the surface of the Moon, wearing a spacesuit, Earth in the background, stars, detailed, cinematic lighting'",
        "NEG_PROMPT = 'photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration'",
        "SEED = 222",
        "",
        "# Generate simple",
        "img_simple = generate_simple_image(pipe, PROMPT, NEG_PROMPT, SEED)",
        "img_simple.save('simple_image.png')",
        "print('Simple image done')",
        "",
        "# Generate advanced",
        "img_advanced = generate_advanced_image(pipe, PROMPT, NEG_PROMPT, SEED, guidance_scale=7.5, num_inference_steps=30)",
        "img_advanced.save('advanced_image.png')",
        "print('Advanced image done')",
        "",
        "# Plot side by side",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 7))",
        "axes[0].imshow(img_simple)",
        "axes[0].set_title('Simple Image (default params)', fontsize=10)",
        "axes[0].axis('off')",
        "axes[1].imshow(img_advanced)",
        "axes[1].set_title('Advanced Image (guidance=7.5, steps=30)', fontsize=10)",
        "axes[1].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('comparison_simple_advanced.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    # --- Experiment: Guidance Scale ---
    cells.append(md("## 5. Eksperimen Guidance Scale"))
    cells.append(md("Gw coba varias nilai guidance scale dari rendah ke tinggi buat liat perbedaannya.\\n"
                     "Nilai rendah biasanya bikin gambar lebih bebas, nilai tinggi lebih strict ikut prompt."))
    cells.append(code([
        "guidance_values = [2, 4, 7.5, 10, 15, 20]",
        "gs_results = []",
        "",
        "for gs in guidance_values:",
        "    img = generate_advanced_image(pipe, PROMPT, NEG_PROMPT, SEED, guidance_scale=gs, num_inference_steps=30)",
        "    img.save(f'guidance_scale_{gs}.png')",
        "    gs_results.append((gs, img))",
        "    print(f'Guidance scale {gs} selesai')",
        "",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))",
        "for i, (gs, img) in enumerate(gs_results):",
        "    r, c = i // 3, i % 3",
        "    axes[r, c].imshow(img)",
        "    axes[r, c].set_title(f'Guidance Scale = {gs}', fontsize=12)",
        "    axes[r, c].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('guidance_scale_comparison.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    cells.append(md("### Observasi Guidance Scale"))
    cells.append(md("**Scale Rendah (2-4):**\\n"
                     "- Gambar keliatan lebih kabur dan kurang detail\\n"
                     "- Kadang muncul objek yang ngga ada di prompt\\n"
                     "- Komposisinya lebih random, warnanya kurang kontras\\n"
                     "- Intinya sih model jadi lebih 'kreatif' karena ngga terlalu dipaksa\\n"
                     "\\n"
                     "**Scale Tinggi (10-20):**\\n"
                     "- Detail lebih tajam dan ngikutin prompt\\n"
                     "- Warnanya lebih kontras, komposisi lebih bagus\\n"
                     "- Tapi kalo kegedean (15+), warnanya jadi overcooked dan muncul artefak\\n"
                     "- Gambar keliatan lebih 'kaku'\\n"
                     "\\n"
                     "**Kesimpulan:** 7.5 itu sweet spotnya. Ngga terlalu rendah, ngga terlalu tinggi."))

    # --- Experiment: Inference Steps ---
    cells.append(md("## 6. Eksperimen Inference Steps"))
    cells.append(md("Step rendah (5-15) vs step tinggi (30-50). Pengaruhnya ke detail dan noise."))
    cells.append(code([
        "step_values = [5, 10, 15, 30, 40, 50]",
        "step_results = []",
        "",
        "for steps in step_values:",
        "    img = generate_advanced_image(pipe, PROMPT, NEG_PROMPT, SEED, guidance_scale=7.5, num_inference_steps=steps)",
        "    img.save(f'inference_steps_{steps}.png')",
        "    step_results.append((steps, img))",
        "    print(f'Steps {steps} selesai')",
        "",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))",
        "for i, (steps, img) in enumerate(step_results):",
        "    r, c = i // 3, i % 3",
        "    axes[r, c].imshow(img)",
        "    axes[r, c].set_title(f'Steps = {steps}', fontsize=12)",
        "    axes[r, c].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('inference_steps_comparison.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    cells.append(md("### Observasi Inference Steps"))
    cells.append(md("**Step Rendah (5-15):**\\n"
                     "- 5 steps: gambar masih kacau, noise dimana-mana, hampir ngga keliatan bentuknya\\n"
                     "- 10 steps: mulai keliatan objek tapi masih burem\\n"
                     "- 15 steps: udah lumayan tapi detail tepi masih kurang\\n"
                     "- Intinya: cepet tapi kualitasnya kurang\\n"
                     "\\n"
                     "**Step Tinggi (30-50):**\\n"
                     "- 30 steps: udah bagus, detail oke, noise minimal\\n"
                     "- 40-50 steps: bedanya tipis banget sama 30 steps (diminishing returns)\\n"
                     "- Gambar lebih stabil dan halus\\n"
                     "\\n"
                     "**Kesimpulan:** 30 steps udah cukup. Lebih dari itu nambah waktu doang tapi hasil hampir sama."))

    # --- Batch Inference ---
    cells.append(md("## 7. Batch Inference (4 gambar, grid 2x2)"))
    cells.append(md("Generate 4 gambar sekaligus pake seed yang beda-beda."))
    cells.append(code([
        "seeds_batch = [42, 123, 256, 777]",
        "batch_imgs = []",
        "",
        "for s in seeds_batch:",
        "    img = generate_advanced_image(pipe, PROMPT, NEG_PROMPT, s, guidance_scale=7.5, num_inference_steps=30)",
        "    batch_imgs.append(img)",
        "    img.save(f'batch_seed_{s}.png')",
        "    print(f'Seed {s} selesai')",
        "",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 12))",
        "for i, (s, img) in enumerate(zip(seeds_batch, batch_imgs)):",
        "    r, c = i // 2, i % 2",
        "    axes[r, c].imshow(img)",
        "    axes[r, c].set_title(f'Seed = {s}', fontsize=11)",
        "    axes[r, c].axis('off')",
        "plt.suptitle('Batch Inference 2x2 Grid', fontsize=14)",
        "plt.tight_layout()",
        "plt.savefig('batch_inference_grid.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    # --- Scheduler Comparison ---
    cells.append(md("## 8. Perbandingan Scheduler"))
    cells.append(md("### 8.1 load\\_scheduler()"))
    cells.append(md("Fungsi buat ganti-ganti scheduler tanpa reload model."))
    cells.append(code([
        "def load_scheduler(pipe, scheduler_name):",
        "    if scheduler_name.lower() == 'euler a':",
        "        sch = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)",
        "    elif scheduler_name.lower() == 'dpm++':",
        "        sch = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)",
        "    elif scheduler_name.lower() == 'ddim':",
        "        sch = DDIMScheduler.from_config(pipe.scheduler.config)",
        "    else:",
        "        raise ValueError(f'Gak kenal scheduler: {scheduler_name}')",
        "    pipe.scheduler = sch",
        "    print(f'Scheduler diganti ke {scheduler_name}')",
        "    return pipe",
    ]))

    cells.append(md("### 8.2 Generate pake 3 Scheduler"))
    cells.append(code([
        "scheduler_list = ['Euler A', 'DPM++', 'DDIM']",
        "sched_results = {}",
        "",
        "for sc in scheduler_list:",
        "    pipe = load_scheduler(pipe, sc)",
        "    img = generate_advanced_image(pipe, PROMPT, NEG_PROMPT, SEED, guidance_scale=7.5, num_inference_steps=30)",
        "    sched_results[sc] = img",
        "    img.save(f'scheduler_{sc.replace(\" \", \"_\")}.png')",
        "    print(f'{sc} selesai')",
        "",
        "# Kembalikan ke scheduler default",
        "pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "for i, (name, img) in enumerate(sched_results.items()):",
        "    axes[i].imshow(img)",
        "    axes[i].set_title(f'{name}', fontsize=12)",
        "    axes[i].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('scheduler_comparison.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    cells.append(md("### Observasi Scheduler"))
    cells.append(md("**Euler A:**\\n"
                     "- Hasilnya lebih bervariasi, ada efek 'ancestral' noise\\n"
                     "- Cocok buat eksplorasi kreatif, tiap run bisa beda\\n"
                     "\\n"
                     "**DPM++:**\\n"
                     "- Detail paling tajam dibanding yang lain\\n"
                     "- Konvergensi cepet, jadi lebih efisien\\n"
                     "- Favorit gw personally\\n"
                     "\\n"
                     "**DDIM:**\\n"
                     "- Paling deterministik, hasilnya konsisten\\n"
                     "- Lebih mulus, transisi halus\\n"
                     "- Cocok kalo butuh reproducible results"))

    # --- Inpainting Setup ---
    cells.append(md("## 9. Inpainting"))
    cells.append(md("### 9.1 Load Model Inpainting"))
    cells.append(code([
        "INPAINT_MODEL_ID = 'runwayml/stable-diffusion-inpainting'",
        "",
        "pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(",
        "    INPAINT_MODEL_ID,",
        "    torch_dtype=torch.float16,",
        "    safety_checker=None,",
        "    requires_safety_checker=False,",
        ")",
        "pipe_inpaint = pipe_inpaint.to(device)",
        "pipe_inpaint.set_progress_bar_config(disable=True)",
        "print('Model inpainting siap')",
    ]))

    cells.append(md("### 9.2 Fungsi inpaint\\_engine()"))
    cells.append(code([
        "def inpaint_engine(pipe, image, mask, prompt, seed=9, steps=30, guidance=7.5):",
        "    generator = torch.Generator(device=device).manual_seed(seed)",
        "    result = pipe(",
        "        prompt=prompt,",
        "        image=image,",
        "        mask_image=mask,",
        "        generator=generator,",
        "        num_inference_steps=steps,",
        "        guidance_scale=guidance,",
        "    )",
        "    return result.images[0]",
    ]))

    cells.append(md("### 9.3 Manual Mask + Inpainting (Broken Satellite)"))
    cells.append(md("Bikin mask hardcode pake trial and error. Seed 9.\\n"
                     "Areanya di langit (pojok kanan atas) biar ditaro broken satellite."))
    cells.append(code([
        "# Load gambar hasil generate tadi",
        "base_img = Image.open('advanced_image.png').convert('RGB').resize((512, 512))",
        "base_img.save('base_inpaint_input.png')",
        "",
        "# Bikin mask manual - trial error dapet posisi",
        "# Coba-coba koordinat sampe dapet area langit yang pas",
        "mask = Image.new('RGB', (512, 512), (0, 0, 0))",
        "draw = ImageDraw.Draw(mask)",
        "draw.ellipse([(350, 60), (460, 200)], fill=(255, 255, 255))",
        "mask.save('manual_mask.png')",
        "",
        "# Convert ke grayscale",
        "mask_gray = mask.convert('L')",
        "",
        "# Prompt buat inpainting",
        "inpaint_prompt = 'A broken satellite drifting in space, damaged solar panels, debris, detailed, cinematic lighting'",
        "",
        "# Jalanin inpainting",
        "img_inpainted = inpaint_engine(",
        "    pipe_inpaint,",
        "    image=base_img,",
        "    mask=mask_gray,",
        "    prompt=inpaint_prompt,",
        "    seed=9,",
        ")",
        "img_inpainted.save('inpainted_broken_satellite.png')",
        "",
        "# Tampilin",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(base_img)",
        "axes[0].set_title('Base Image', fontsize=12)",
        "axes[0].axis('off')",
        "axes[1].imshow(mask, cmap='gray')",
        "axes[1].set_title('Manual Mask', fontsize=12)",
        "axes[1].axis('off')",
        "axes[2].imshow(img_inpainted)",
        "axes[2].set_title('Inpainted: Broken Satellite', fontsize=12)",
        "axes[2].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('inpainting_result.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    # --- Segmentation Masking ---
    cells.append(md("## 10. Segmentation-Based Masking"))
    cells.append(md("Pake threshold warna buat deteksi area langit secara otomatis."))
    cells.append(code([
        "def create_sky_mask(image, threshold=80):",
        "    img_arr = np.array(image)",
        "    # Deteksi warna biru langit",
        "    lower = np.array([100 - threshold, 150 - threshold, 255 - threshold])",
        "    upper = np.array([100 + threshold, 150 + threshold, 255])",
        "    mask_arr = np.all((img_arr >= lower) & (img_arr <= upper), axis=-1)",
        "    return Image.fromarray((mask_arr * 255).astype(np.uint8))",
        "",
        "sky_mask = create_sky_mask(base_img, threshold=80)",
        "sky_mask = sky_mask.resize((512, 512))",
        "sky_mask.save('segmentation_sky_mask.png')",
        "",
        "img_seg = inpaint_engine(",
        "    pipe_inpaint,",
        "    image=base_img,",
        "    mask=sky_mask.convert('L'),",
        "    prompt='A broken satellite with solar panels, drifting in orbit, detailed',",
        "    seed=9,",
        ")",
        "img_seg.save('segmentation_inpainted.png')",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(base_img)",
        "axes[0].set_title('Original', fontsize=12)",
        "axes[0].axis('off')",
        "axes[1].imshow(sky_mask, cmap='gray')",
        "axes[1].set_title('Sky Mask (Auto)', fontsize=12)",
        "axes[1].axis('off')",
        "axes[2].imshow(img_seg)",
        "axes[2].set_title('Result', fontsize=12)",
        "axes[2].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('segmentation_inpainting_result.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    # --- Outpainting ---
    cells.append(md("## 11. Outpainting"))
    cells.append(md("### 11.1 prepare\\_outpainting()"))
    cells.append(md("Fungsi buat perluas kanvas ke satu arah."))
    cells.append(code([
        "def prepare_outpainting(image, direction, expand=128):",
        "    w, h = image.size",
        "    if direction == 'right':",
        "        new = Image.new('RGB', (w + expand, h), (0, 0, 0))",
        "        new.paste(image, (0, 0))",
        "        m = Image.new('L', (w + expand, h), 0)",
        "        m.paste(Image.new('L', (expand, h), 255), (w, 0))",
        "    elif direction == 'left':",
        "        new = Image.new('RGB', (w + expand, h), (0, 0, 0))",
        "        new.paste(image, (expand, 0))",
        "        m = Image.new('L', (w + expand, h), 0)",
        "        m.paste(Image.new('L', (expand, h), 255), (0, 0))",
        "    elif direction == 'top':",
        "        new = Image.new('RGB', (w, h + expand), (0, 0, 0))",
        "        new.paste(image, (0, expand))",
        "        m = Image.new('L', (w, h + expand), 0)",
        "        m.paste(Image.new('L', (w, expand), 255), (0, 0))",
        "    elif direction == 'bottom':",
        "        new = Image.new('RGB', (w, h + expand), (0, 0, 0))",
        "        new.paste(image, (0, 0))",
        "        m = Image.new('L', (w, h + expand), 0)",
        "        m.paste(Image.new('L', (w, expand), 255), (0, h))",
        "    else:",
        "        raise ValueError('Invalid direction')",
        "    return new, m",
    ]))

    cells.append(md("### 11.2 Outpainting ke Kanan"))
    cells.append(md("Pake gambar hasil inpainting sebagai input, perluas ke kanan."))
    cells.append(code([
        "out_base = img_inpainted.resize((512, 512))",
        "",
        "exp_img, out_mask = prepare_outpainting(out_base, 'right', 128)",
        "exp_img.save('outpaint_expanded.png')",
        "out_mask.save('outpaint_mask.png')",
        "",
        "out_result = pipe_inpaint(",
        "    prompt=PROMPT + ', extended cityscape',",
        "    image=exp_img,",
        "    mask_image=out_mask,",
        "    generator=torch.Generator(device=device).manual_seed(42),",
        "    num_inference_steps=30,",
        "    guidance_scale=7.5,",
        ").images[0]",
        "out_result.save('outpainting_result.png')",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
        "axes[0].imshow(out_base)",
        "axes[0].set_title('Original', fontsize=12)",
        "axes[0].axis('off')",
        "axes[1].imshow(exp_img)",
        "axes[1].set_title('Expanded + Mask', fontsize=12)",
        "axes[1].axis('off')",
        "axes[2].imshow(out_result)",
        "axes[2].set_title('Outpainting Result', fontsize=12)",
        "axes[2].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('outpainting_demo.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    # --- Zoom Out ---
    cells.append(md("## 12. Zoom-Out Outpainting"))
    cells.append(md("Perluas gambar ke semua arah secara bertahap."))
    cells.append(code([
        "def zoom_out(pipe, image, expand=64, prompt=None, seed=42):",
        "    if prompt is None:",
        "        prompt = PROMPT",
        "    result = image.copy()",
        "    for direction in ['left', 'right', 'top', 'bottom']:",
        "        exp, m = prepare_outpainting(result, direction, expand)",
        "        result = pipe(",
        "            prompt=prompt + ', wide angle view',",
        "            image=exp,",
        "            mask_image=m,",
        "            generator=torch.Generator(device=device).manual_seed(seed),",
        "            num_inference_steps=30,",
        "            guidance_scale=7.5,",
        "        ).images[0]",
        "        seed += 1",
        "    return result",
        "",
        "zoom_img = zoom_out(pipe_inpaint, img_inpainted.resize((384, 384)), expand=64)",
        "zoom_img.save('zoom_out_result.png')",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 7))",
        "axes[0].imshow(img_inpainted.resize((384, 384)))",
        "axes[0].set_title('Before', fontsize=12)",
        "axes[0].axis('off')",
        "axes[1].imshow(zoom_img)",
        "axes[1].set_title('After Zoom-Out', fontsize=12)",
        "axes[1].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('zoom_out_comparison.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    # --- Two Stage ---
    cells.append(md("## 13. Two-Stage Generation (Base + Refiner)"))
    cells.append(md("Stage 1: base pipeline sampe 80% denoising.\\n"
                     "Stage 2: refiner pake Img2Img buat 20% sisanya."))
    cells.append(code([
        "pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(",
        "    MODEL_ID,",
        "    torch_dtype=torch.float16,",
        "    safety_checker=None,",
        "    requires_safety_checker=False,",
        ").to(device)",
        "pipe_img2img.set_progress_bar_config(disable=True)",
        "print('Img2Img pipeline siap')",
    ]))

    cells.append(code([
        "def two_stage(pipe_base, pipe_refiner, prompt, neg_prompt, seed, guidance=7.5, steps=50):",
        "    gen = torch.Generator(device=device).manual_seed(seed)",
        "",
        "    # Stage 1: Base - dapetin latent di 80%",
        "    out = pipe_base(",
        "        prompt=prompt,",
        "        negative_prompt=neg_prompt,",
        "        generator=gen,",
        "        guidance_scale=guidance,",
        "        num_inference_steps=steps,",
        "        denoising_end=0.8,",
        "        output_type='latent',",
        "    )",
        "    latents = out.images  # [1, 4, 64, 64]",
        "",
        "    # Decode latent ke image tensor [0,1]",
        "    with torch.no_grad():",
        "        scaled = latents / pipe_base.vae.config.scaling_factor",
        "        img_t = pipe_base.vae.decode(scaled, return_dict=False)[0]  # [1, 3, 512, 512]",
        "        img_t = (img_t / 2 + 0.5).clamp(0, 1)",
        "",
        "    # Stage 2: Refine 20% sisanya",
        "    refined = pipe_refiner(",
        "        prompt=prompt,",
        "        negative_prompt=neg_prompt,",
        "        image=img_t,",
        "        generator=gen,",
        "        guidance_scale=guidance,",
        "        num_inference_steps=steps,",
        "        denoising_start=0.8,",
        "    ).images[0]",
        "",
        "    return refined",
    ]))

    cells.append(code([
        "img_twostage = two_stage(pipe, pipe_img2img, PROMPT, NEG_PROMPT, 222)",
        "img_twostage.save('twostage_result.png')",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 7))",
        "axes[0].imshow(img_advanced)",
        "axes[0].set_title('Standard (30 steps)', fontsize=11)",
        "axes[0].axis('off')",
        "axes[1].imshow(img_twostage)",
        "axes[1].set_title('Two-Stage (50 steps)', fontsize=11)",
        "axes[1].axis('off')",
        "plt.tight_layout()",
        "plt.savefig('twostage_comparison.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ]))

    cells.append(md("### Observasi Two-Stage"))
    cells.append(md("Hasilnya lebih detail dibanding standard generation.\\n"
                     "Tapi ya konsekuensinya: waktu generate jadi 2x lipat dan VRAM lebih boros.\\n"
                     "Cocok buat penggunaan yang bener-bener butuh kualitas tinggi."))

    # --- Cleanup ---
    cells.append(md("## 14. Cleanup"))
    cells.append(code([
        "gc.collect()",
        "torch.cuda.empty_cache()",
        "print('Memory udah dibersihin. Selesai!')",
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


# ============================================================
# STREAMLIT NOTEBOOK
# ============================================================
def build_streamlit():
    cells = []

    cells.append(md("# Streamlit App - Generative Image Suite"))
    cells.append(md("Aplikasi web sederhana pake Streamlit buat text-to-image, inpainting, sama outpainting.\\n---"))

    cells.append(md("## 1. Install"))
    cells.append(code([
        "!pip install streamlit diffusers transformers accelerate torch pillow numpy streamlit-drawable-canvas",
    ]))

    cells.append(md("## 2. Kode Aplikasi"))
    cells.append(md("jalanin cell ini buat generate file app.py, terus jalanin `streamlit run app.py` di terminal."))
    cells.append(code([
        "%%writefile app.py",
        "import streamlit as st",
        "import torch",
        "import gc",
        "import numpy as np",
        "from PIL import Image, ImageDraw, ImageOps",
        "import io",
        "",
        "from diffusers import (",
        "    StableDiffusionPipeline,",
        "    StableDiffusionInpaintPipeline,",
        "    DDIMScheduler,",
        "    DPMSolverMultistepScheduler,",
        "    EulerAncestralDiscreteScheduler,",
        ")",
        "",
        "st.set_page_config(page_title='Generative Image Suite', page_icon=':art:', layout='wide')",
        "",
        "MODEL_ID = 'runwayml/stable-diffusion-v1-5'",
        "INPAINT_MODEL = 'runwayml/stable-diffusion-inpainting'",
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "",
        "@st.cache_resource",
        "def load_models():",
        "    pipe = StableDiffusionPipeline.from_pretrained(",
        "        MODEL_ID, torch_dtype=torch.float16,",
        "        safety_checker=None, requires_safety_checker=False",
        "    ).to(DEVICE)",
        "    pipe.set_progress_bar_config(disable=True)",
        "",
        "    pipe_i = StableDiffusionInpaintPipeline.from_pretrained(",
        "        INPAINT_MODEL, torch_dtype=torch.float16,",
        "        safety_checker=None, requires_safety_checker=False",
        "    ).to(DEVICE)",
        "    pipe_i.set_progress_bar_config(disable=True)",
        "    return pipe, pipe_i",
        "",
        "def ganti_scheduler(pipe, name):",
        "    if name == 'Euler A':",
        "        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)",
        "    elif name == 'DPM++':",
        "        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)",
        "    elif name == 'DDIM':",
        "        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)",
        "    return pipe",
        "",
        "def bersihin_memory():",
        "    gc.collect()",
        "    if torch.cuda.is_available():",
        "        torch.cuda.empty_cache()",
        "    st.success('Memory cleared!')",
        "",
        "def generate(pipe, prompt, neg, seed, gs, steps, n):",
        "    hasil = []",
        "    for i in range(n):",
        "        gen = torch.Generator(device=DEVICE).manual_seed(seed + i)",
        "        out = pipe(",
        "            prompt=prompt, negative_prompt=neg,",
        "            generator=gen, guidance_scale=gs, num_inference_steps=steps",
        "        )",
        "        hasil.append(out.images[0])",
        "    return hasil",
        "",
        "def buat_grid(images, cols=2):",
        "    rows = (len(images) + cols - 1) // cols",
        "    w, h = images[0].size",
        "    grid = Image.new('RGB', (cols * w, rows * h))",
        "    for i, img in enumerate(images):",
        "        r, c = i // cols, i % cols",
        "        grid.paste(img.resize((w, h)), (c * w, r * h))",
        "    return grid",
        "",
        "# === UI ====",
        "st.title('Generative Image Suite')",
        "st.markdown('Aplikasi buat generate dan edit gambar pake Stable Diffusion 1.5')",
        "st.markdown('---')",
        "",
        "with st.spinner('Loading models... (butuh beberapa menit)'):",
        "    pipe_t2i, pipe_inpaint = load_models()",
        "st.success('Models ready!')",
        "",
        "tab1, tab2 = st.tabs(['Text-to-Image', 'Inpainting / Outpainting'])",
        "",
        "# === TAB 1: TEXT-TO-IMAGE ===",
        "with tab1:",
        "    col1, col2 = st.columns([1, 2])",
        "    with col1:",
        "        st.subheader('Settings')",
        "        prompt = st.text_area('Prompt',",
        "            'An astronaut standing on the surface of the Moon, wearing a spacesuit, Earth in the background, stars, detailed, cinematic lighting',",
        "            height=80)",
        "        neg = st.text_area('Negative Prompt',",
        "            'photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration',",
        "            height=80)",
        "",
        "        col_s1, col_s2 = st.columns(2)",
        "        with col_s1:",
        "            gs = st.slider('Guidance Scale', 1.0, 20.0, 7.5, 0.5)",
        "        with col_s2:",
        "            steps = st.slider('Inference Steps', 5, 100, 30, 5)",
        "",
        "        seed = st.number_input('Seed', value=222, step=1)",
        "        num = st.number_input('Jumlah Gambar', min_value=1, max_value=4, value=1)",
        "        sched = st.selectbox('Scheduler', ['Euler A', 'DPM++', 'DDIM'])",
        "",
        "        col_b1, col_b2 = st.columns(2)",
        "        with col_b1:",
        "            btn_gen = st.button('Generate', type='primary', use_container_width=True)",
        "        with col_b2:",
        "            if st.button('Clear Memory', use_container_width=True):",
        "                bersihin_memory()",
        "",
        "    with col2:",
        "        st.subheader('Hasil')",
        "        if btn_gen:",
        "            with st.spinner('Generating...'):",
        "                pipe_t2i = ganti_scheduler(pipe_t2i, sched)",
        "                images = generate(pipe_t2i, prompt, neg, int(seed), gs, steps, int(num))",
        "                if len(images) == 1:",
        "                    st.image(images[0], use_container_width=True)",
        "                else:",
        "                    st.image(buat_grid(images, 2), use_container_width=True)",
        "                st.session_state['gen_images'] = images",
        "                st.session_state['last_prompt'] = prompt",
        "",
        "# === TAB 2: INPAINTING / OUTPAINTING ===",
        "with tab2:",
        "    st.subheader('Edit Gambar')",
        "",
        "    if 'gen_images' in st.session_state and len(st.session_state['gen_images']) > 0:",
        "        opts = [f'Image {i+1}' for i in range(len(st.session_state['gen_images']))]",
        "        idx = st.selectbox('Pilih gambar', range(len(opts)), format_func=lambda x: opts[x])",
        "        src = st.session_state['gen_images'][idx].resize((512, 512))",
        "    else:",
        "        st.warning('Generate dulu di tab Text-to-Image atau upload gambar')",
        "        up = st.file_uploader('Upload', type=['png', 'jpg'])",
        "        if up:",
        "            src = Image.open(up).convert('RGB').resize((512, 512))",
        "            st.session_state['gen_images'] = [src]",
        "            idx = 0",
        "        else:",
        "            st.stop()",
        "",
        "    col_e1, col_e2 = st.columns([1, 1])",
        "    with col_e1:",
        "        st.image(src, caption='Source', use_container_width=True)",
        "        st.subheader('Inpainting')",
        "        inp_prompt = st.text_input('Prompt inpainting', 'A broken satellite in space, damaged, detailed')",
        "",
        "        try:",
        "            from streamlit_drawable_canvas import st_canvas",
        "            canvas = st_canvas(",
        "                fill_color='rgba(255,255,255,1.0)',",
        "                stroke_width=25, stroke_color='rgba(255,255,255,1.0)',",
        "                background_image=src, height=512, width=512,",
        "                drawing_mode='freedraw', key='canvas'",
        "            )",
        "            col_btn, col_clr = st.columns(2)",
        "            with col_btn:",
        "                btn_inp = st.button('Run Inpainting', type='primary', use_container_width=True)",
        "            with col_clr:",
        "                if st.button('Clear Memory', use_container_width=True):",
        "                    bersihin_memory()",
        "        except:",
        "            st.info('Drawable canvas unavailable, pake mode rectangle')",
        "            mx = st.slider('X', 0, 512, 350)",
        "            my = st.slider('Y', 0, 512, 60)",
        "            mw = st.slider('Width', 10, 200, 110)",
        "            mh = st.slider('Height', 10, 200, 140)",
        "            btn_inp = st.button('Run Inpainting', type='primary')",
        "            canvas = None",
        "",
        "    with col_e2:",
        "        st.subheader('Result')",
        "        if btn_inp:",
        "            with st.spinner('Inpainting...'):",
        "                if canvas is not None and canvas.image_data is not None:",
        "                    arr = canvas.image_data[:, :, 0].astype(np.uint8)",
        "                    mask_img = Image.fromarray(((arr > 0) * 255).astype(np.uint8), mode='L')",
        "                else:",
        "                    mask_img = Image.new('L', (512, 512), 0)",
        "                    d = ImageDraw.Draw(mask_img)",
        "                    d.rectangle([mx, my, mx+mw, my+mh], fill=255)",
        "",
        "                gen = torch.Generator(device=DEVICE).manual_seed(9)",
        "                hasil = pipe_inpaint(",
        "                    prompt=inp_prompt,",
        "                    image=src.resize((512, 512)),",
        "                    mask_image=mask_img.resize((512, 512)),",
        "                    generator=gen, num_inference_steps=30, guidance_scale=7.5",
        "                ).images[0]",
        "                st.image(hasil, use_container_width=True)",
        "                st.session_state['inpainted'] = hasil",
        "",
        "        # Zoom Out",
        "        st.markdown('---')",
        "        st.subheader('Zoom-Out')",
        "        if 'inpainted' in st.session_state:",
        "            base_zoom = st.session_state['inpainted']",
        "        elif 'gen_images' in st.session_state:",
        "            base_zoom = st.session_state['gen_images'][0]",
        "        else:",
        "            base_zoom = src",
        "",
        "        expand = st.slider('Expand (px)', 32, 256, 96, 16)",
        "        if st.button('Run Zoom-Out', type='primary'):",
        "            with st.spinner('Zooming out...'):",
        "                result = base_zoom.resize((384, 384))",
        "                for direction in ['left', 'right', 'top', 'bottom']:",
        "                    w, h = result.size",
        "                    e = int(expand)",
        "                    if direction == 'right':",
        "                        new_img = Image.new('RGB', (w+e, h), (0,0,0))",
        "                        new_img.paste(result, (0,0))",
        "                        m = Image.new('L', (w+e, h), 0)",
        "                        ImageDraw.Draw(m).rectangle([w, 0, w+e, h], fill=255)",
        "                    elif direction == 'left':",
        "                        new_img = Image.new('RGB', (w+e, h), (0,0,0))",
        "                        new_img.paste(result, (e,0))",
        "                        m = Image.new('L', (w+e, h), 0)",
        "                        ImageDraw.Draw(m).rectangle([0, 0, e, h], fill=255)",
        "                    elif direction == 'top':",
        "                        new_img = Image.new('RGB', (w, h+e), (0,0,0))",
        "                        new_img.paste(result, (0,e))",
        "                        m = Image.new('L', (w, h+e), 0)",
        "                        ImageDraw.Draw(m).rectangle([0, 0, w, e], fill=255)",
        "                    else:",
        "                        new_img = Image.new('RGB', (w, h+e), (0,0,0))",
        "                        new_img.paste(result, (0,0))",
        "                        m = Image.new('L', (w, h+e), 0)",
        "                        ImageDraw.Draw(m).rectangle([0, h, w, h+e], fill=255)",
        "",
        "                    prompt_zoom = st.session_state.get('last_prompt', 'moon surface, stars')",
        "                    result = pipe_inpaint(",
        "                        prompt=prompt_zoom, image=new_img, mask_image=m,",
        "                        generator=torch.Generator(device=DEVICE).manual_seed(42),",
        "                        num_inference_steps=30, guidance_scale=7.5",
        "                    ).images[0]",
        "                st.image(result, use_container_width=True)",
        "",
        "st.markdown('---')",
        "st.caption('Generative Image Suite - BFGAI Submission')",
    ]))

    cells.append(md("## 3. Jalankan Streamlit"))
    cells.append(md("Setelah cell di atas di-run, bakal ada file `app.py`.\\n"
                     "Buka terminal dan jalanin:\\n"
                     "```bash\\n"
                     "streamlit run app.py\\n"
                     "```"))

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


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import os
    outdir = os.path.dirname(os.path.abspath(__file__))

    for name, builder in [
        ('Pipeline_submission_BFGAI_ImageGeneration.ipynb', build_pipeline),
        ('Streamlit_submission_BFGAI_ImageGeneration.ipynb', build_streamlit),
    ]:
        path = os.path.join(outdir, name)
        with open(path, 'w') as f:
            json.dump(builder(), f, indent=1, ensure_ascii=False)
        print(f'Created: {path}')

    print('\\nDone!')
