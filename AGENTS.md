# Project Memory untuk ImageGeneration (BFGAI Submission)

## Prompt yang digunakan
- **PROMPT**: `"An astronaut standing on the surface of the Moon, wearing a spacesuit, Earth in the background, stars, detailed, cinematic lighting"`
- **NEG**: `"photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration"`
- **SEED**: 222 (untuk kriteria 1), 9 (untuk inpainting)

## Kriteria 1: Text-to-Image
- `generate_simple_image()`: prompt, negative_prompt, seed
- `generate_advanced_image()`: prompt, negative_prompt, seed, guidance_scale, num_inference_steps
- Kedua fungsi menggunakan PROMPT yang SAMA
- Output: `simple_image.png` dan `advanced_image.png`

## Kriteria 2: Image-to-Image
- Base image untuk inpainting adalah `advanced_image.png` (astronaut di bulan)
- `inpaint_engine()` menggunakan model `runwayml/stable-diffusion-inpainting`
- Mask dibuat manual (hardcode) dengan trial and error
- Seed: 9, prompt inpainting: "A broken satellite drifting in space, damaged solar panels, debris, detailed, cinematic"
- Output inpainting: `inpainted_satellite.png`
- Two-Stage Generation:
  - Stage 1: Base pipeline dengan `denoising_end=0.8`, output_type='latent'
  - Stage 2: Img2Img pipeline dengan `denoising_start=0.8`

## Kriteria 3: Streamlit
- Menggunakan template `[Template]_Streamlit_submission_BFGAI_Nama_siswa.ipynb`
- File logic.py berisi implementasi generate_image, flush_memory, set_scheduler, run_inpainting, prepare_outpainting
- File app.py menggunakan Streamlit dengan tab Generate dan Edit

## File penting
- `Pipeline_submission_BFGAI_ImageGeneration.ipynb` - Notebook eksperimen
- `Streamlit_submission_BFGAI_ImageGeneration.ipynb` - Notebook Streamlit (mengikuti template)
- `app.py` - Aplikasi Streamlit standalone
- `generate_submission.py` - Generator untuk membuat notebook
- `gen_pipeline_v2.py` - Generator pipeline versi kedua
- `requirements.txt` - Dependencies

## Catatan
- Semua gambar tersimpan sebagai file PNG di folder project
- Gunakan `torch.float16` untuk menghemat VRAM
- Pipeline menggunakan `safety_checker=None`
- Tidak perlu `.to(device)` jika menggunakan `enable_sequential_cpu_offload()`
