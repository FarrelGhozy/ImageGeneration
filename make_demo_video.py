#!/usr/bin/env python3
"""Generate demo video (1-3 min) from pipeline images."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, time

OUTPUT = 'demo_video.mp4'
FPS = 1

def make_text_frame(lines, size=(1280, 720)):
    img = Image.new('RGB', size, (20, 20, 35))
    draw = ImageDraw.Draw(img)
    try:
        font_t = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 42)
        font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
        font_xs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    except:
        font_t = ImageFont.load_default(); font_s = font_t; font_xs = font_t

    y_start = 100
    for text, level in lines:
        font = {'t': font_t, 's': font_s, 'xs': font_xs}[level]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        color = {'t': (220, 220, 240), 's': (190, 190, 210), 'xs': (150, 150, 170)}[level]
        draw.text(((size[0]-tw)//2, y_start), text, fill=color, font=font)
        y_start += th + 10
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def make_img_frame(path, label="", size=(1280, 720)):
    if not os.path.exists(path):
        return make_text_frame([(f"File not found: {path}", "s")], size)
    img = Image.open(path).convert('RGB')
    iw, ih = img.size
    target_w, target_h = size[0] - 100, size[1] - 100
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, (20, 20, 35))
    x, y = (size[0] - new_w) // 2, (size[1] - new_h) // 2
    canvas.paste(img, (x, y))
    if label:
        draw = ImageDraw.Draw(canvas)
        try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2]-bbox[0]
        draw.text(((size[0]-lw)//2, size[1]-30), label, fill=(200, 200, 200), font=font)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def add_sec(n, frame):
    return [frame] * (n * FPS)

CWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(CWD)

scene = []

# Scene 1: Title
frame = make_text_frame([("Image Generation Pipeline", "t"),
    ("Stable Diffusion 1.5 - Pipeline + Streamlit", "s"),
    ("BFGAI - Dicoding Submission", "xs")])
scene += add_sec(5, frame)

# Scene 2: Overview
frame = make_text_frame([("Pipeline Overview", "t"),
    ("", "xs"),
    ("Model: runwayml/stable-diffusion-v1-5", "s"),
    ("Inpainting: runwayml/stable-diffusion-inpainting", "s"),
    ("Device: NVIDIA RTX 4060 (CUDA)", "s"),
    ("", "xs"),
    ("7 Experiments + Inpainting + Outpainting + ZoomOut + Two-Stage", "s"),
    ("", "xs"),
    ("Level: Advanced", "t")])
scene += add_sec(6, frame)

# Scenes: Pipeline Results
scenes_data = [
    ("simple_image.png", "Simple Generation (2D Cartoon style)", 6),
    ("advanced_image.png", "Advanced Generation (Realistic, 30 steps, GS 7.5)", 6),
    ("comparison_simple_advanced.png", "Simple vs Advanced - Different prompt styles", 8),
    ("guidance_scale_comparison.png", "Guidance Scale: 2, 4, 7.5, 10, 15, 20", 8),
    ("inference_steps_comparison.png", "Inference Steps: 5, 10, 15, 30, 40, 50", 8),
    ("batch_inference_grid.png", "Batch Inference - Seeds: 42, 123, 256, 777", 8),
    ("scheduler_comparison.png", "Scheduler: Euler A vs DPM++ vs DDIM", 8),
    ("inpainting_result.png", "Inpainting - Manual mask + broken satellite repair", 8),
    ("segmentation_result.png", "Segmentation Mask - Color-based sky detection", 8),
    ("outpainting_result.png", "Outpainting - Expand image beyond boundaries", 6),
    ("zoom_out_comparison.png", "Zoom Out - Multi-directional expansion", 8),
    ("twostage_comparison.png", "Two-Stage Pipeline - Base + Refinement", 8),
]

for path, label, dur in scenes_data:
    frame = make_img_frame(path, label)
    scene += add_sec(dur, frame)

# Streamlit Demo Section
frame = make_text_frame([("Streamlit Demo Application", "t"),
    ("", "xs"),
    ("Interactive Web UI for Image Generation", "s"),
    ("Text-to-Image, Inpainting, and Parameter Tuning", "s")])
scene += add_sec(5, frame)

strm_scenes = [
    ("strm_simple.png", "Streamlit - Basic Text-to-Image generation", 5),
    ("strm_advanced.png", "Streamlit - Advanced Generation with parameter controls", 5),
    ("strm_inpaint.png", "Streamlit - Inpainting Editor with mask tools", 5),
]

for path, label, dur in strm_scenes:
    frame = make_img_frame(path, label)
    scene += add_sec(dur, frame)

# Final Scene
frame = make_text_frame([("Terima Kasih!", "t"),
    ("", "xs"),
    ("Farrel Ghozy - BFGAI Dicoding", "s"),
    ("All images generated with Stable Diffusion 1.5", "xs")])
scene += add_sec(5, frame)

# Write video
print(f"Total frames: {len(scene)} ({len(scene)/FPS:.0f}s at {FPS}fps)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT, fourcc, FPS, (1280, 720))
for frame in scene:
    out.write(frame)
out.release()
print(f"✅ Video saved: {OUTPUT} ({os.path.getsize(OUTPUT)/1024**2:.1f} MB)")
