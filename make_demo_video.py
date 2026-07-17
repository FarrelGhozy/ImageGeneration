#!/usr/bin/env python3
"""Generate demo video (1-5 min) from pipeline images + streamlit screenshots."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

BASE = '/home/fazy/Documents/DICODING/DIGDAYA/ImageGeneration'
FPS = 1

def make_text_frame(lines, size=(1280, 720)):
    img = Image.new('RGB', size, (20, 20, 35))
    draw = ImageDraw.Draw(img)
    try:
        font_t = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 42)
        font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
        font_xs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except:
        font_t = ImageFont.load_default(); font_s = font_t; font_xs = font_t

    y_start = 120
    for i, (text, level) in enumerate(lines):
        font = {'t': font_t, 's': font_s, 'xs': font_xs}[level]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        color = {'t': (220, 220, 240), 's': (190, 190, 210), 'xs': (150, 150, 170)}[level]
        draw.text(((size[0]-tw)//2, y_start), text, fill=color, font=font)
        y_start += th + 8
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def make_img_frame(path, label="", size=(1280, 720)):
    img = Image.open(path).convert('RGB')
    iw, ih = img.size
    target_w, target_h = size[0] - 100, size[1] - 100
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('RGB', size, (20, 20, 35))
    x, y = (size[0] - new_w) // 2, (size[1] - new_h) // 2
    canvas.paste(img, (x, y))
    if label:
        draw = ImageDraw.Draw(canvas)
        try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2]-bbox[0]
        draw.text(((size[0]-lw)//2, size[1]-40), label, fill=(200, 200, 200), font=font)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def add_sec(n): return [None] * (n * FPS)

scene = []

# ---- SCENE 1: Title (5s) ----
scene += [make_text_frame([("Generative Image Suite", "t"),
    ("Stable Diffusion 1.5 Pipeline & Streamlit App", "s"),
    ("BFGAI - Image Generation Submission", "xs")])] + add_sec(5)

# ---- SCENE 2: Overview (8s) ----
scene += [make_text_frame([("Overview", "t"),
    ("", "xs"),
    ("Model: runwayml/stable-diffusion-v1-5", "s"),
    ("Inpainting: runwayml/stable-diffusion-inpainting", "s"),
    ("", "xs"),
    ("Device: NVIDIA RTX 2050 (4GB VRAM)", "s"),
    ("Optimized for low-memory inference", "s"),
    ("", "xs"),
    ("Level: Advanced", "t")])] + add_sec(8)

# ---- SCENE 3-14: Pipeline Results ----
scenes_data = [
    ("simple_image.png", "Basic Generation - Default params", 8),
    ("advanced_image.png", "Advanced Generation - GS 7.5, 30 steps, Seed 222", 8),
    ("comparison_simple_advanced.png", "Simple vs Advanced - Parameter tuning comparison", 10),
    ("guidance_scale_comparison.png", "Guidance Scale: 2, 4, 7.5, 10, 15, 20", 10),
    ("steps_comparison.png", "Inference Steps: 5, 10, 15, 30, 40, 50", 10),
    ("batch_grid.png", "Batch Inference Grid - 4 seeds (42, 123, 256, 777)", 10),
    ("scheduler_comparison.png", "Scheduler: Euler A vs DPM++ vs DDIM", 10),
    ("inpainting_result.png", "Inpainting - Manual mask + broken satellite", 10),
    ("segmentation_result.png", "Segmentation Masking - Color-based sky mask", 10),
    ("outpainting_demo.png", "Outpainting - Extend canvas to the right", 10),
    ("zoom_out_compare.png", "Zoom-Out Outpainting - 4-direction expansion", 10),
    ("twostage_compare.png", "Two-Stage Generation - T2I latent + Img2Img refine", 10),
]

for fname, label, dur in scenes_data:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f"SKIP: {fname}")
        continue
    scene += [make_img_frame(fpath, label)] + add_sec(dur)

# ---- SCENE 15: Streamlit App (10s) ----
scene += [make_text_frame([("Streamlit Application", "t"),
    ("", "xs"),
    ("Tab 1: Text-to-Image", "s"),
    ("  Prompt, Neg Prompt, GS, Steps, Seed x4", "xs"),
    ("  Scheduler selection (Euler A / DPM++ / DDIM)", "xs"),
    ("", "xs"),
    ("Tab 2: Inpainting / Outpainting", "s"),
    ("  Drawable mask canvas", "xs"),
    ("  Inpainting with custom prompt", "xs"),
    ("  Zoom-Out outpainting (4 directions)", "xs"),
    ("", "xs"),
    ("Memory management: Clear Memory button", "xs")])] + add_sec(10)

# ---- SCENE 16: How to Run (8s) ----
scene += [make_text_frame([("How to Run", "t"),
    ("", "xs"),
    ("streamlit run app.py", "s"),
    ("", "xs"),
    ("Requirements in requirements.txt", "xs"),
    ("GPU with CUDA recommended", "xs"),
    ("CPU fallback works but slow", "xs")])] + add_sec(8)

# ---- SCENE 17: End (4s) ----
scene += [make_text_frame([("Demo Complete", "t"),
    ("", "xs"),
    ("Terima Kasih", "s"),
    ("BFGAI Image Generation Submission 2026", "xs")])] + add_sec(4)

# Write video
out_path = os.path.join(BASE, 'demo_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, FPS, (1280, 720))
last_frame = None
for f in scene:
    if f is None:
        out.write(last_frame if last_frame is not None else scene[0])
    else:
        out.write(f)
        last_frame = f
out.release()

total_dur = len(scene) / FPS
print(f"Video: {out_path}")
print(f"Duration: {total_dur:.0f}s ({total_dur/60:.1f} min)")
