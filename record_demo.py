#!/usr/bin/env python3
"""Record demo video of Streamlit app using Xvfb + Chromium + ffmpeg"""
import os, sys, time, signal, subprocess, threading
from pyvirtualdisplay import Display

os.environ['DISPLAY'] = ':99'
os.environ['TMPDIR'] = '/home/fazy/bfgai_tmp'
os.environ['LD_LIBRARY_PATH'] = '/home/fazy/bfgai_venv/nvidia_libs'
BASE_DIR = '/home/fazy/Documents/DICODING/DIGDAYA/ImageGeneration'
VENV_PYTHON = '/home/fazy/bfgai_venv/bin/python3'
VENV_ACTIVATE = 'source /home/fazy/bfgai_venv/bin/activate'

procs = []

def cleanup():
    for p in procs:
        try: p.terminate(); p.wait(timeout=5)
        except: pass
    subprocess.run(['pkill', '-f', 'chromium'], capture_output=True)
    subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)

def sh(cmd, timeout=30, env=None):
    e = os.environ.copy()
    e.update(env or {})
    r = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True,
                       timeout=timeout, env=e)
    if r.returncode != 0:
        print(f"WARN: '{cmd[:80]}' -> {r.stderr[:200]}")
    return r.stdout, r.stderr, r.returncode

# Cleanup old
cleanup()
time.sleep(1)

# 1. Start Xvfb
print("Starting Xvfb...")
disp = Display(visible=True, size=(1280, 720), color_depth=24)
disp.start()
print(f"Xvfb on :{disp.display}")

os.environ['DISPLAY'] = f':{disp.display}'

# 2. Start Streamlit
print("Starting Streamlit...")
streamlit_cmd = f'cd {BASE_DIR} && {VENV_ACTIVATE} && TMPDIR=/home/fazy/bfgai_tmp LD_LIBRARY_PATH=/home/fazy/bfgai_venv/nvidia_libs streamlit run app.py --server.port 8504 --server.headless true'
p_streamlit = subprocess.Popen(['bash', '-c', streamlit_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
procs.append(p_streamlit)
time.sleep(5)

# 3. Start ffmpeg recording
print("Starting ffmpeg recording...")
output_path = os.path.join(BASE_DIR, 'demo_recording.mp4')
ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-f', 'x11grab', '-framerate', '15',
    '-video_size', '1280x720',
    '-i', f':{disp.display}.0+0,0',
    '-c:v', 'libx264', '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p', '-t', '180',
    output_path
]
p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
procs.append(p_ffmpeg)

# Wait for Streamlit to fully load (models take time)
print("Waiting for models to load (120s)...")
time.sleep(120)

# 4. Start Chromium
print("Starting Chromium...")
chromium_cmd = [
    'chromium-browser',
    '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
    '--kiosk', '--incognito',
    'http://localhost:8504'
]
p_chrome = subprocess.Popen(chromium_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
procs.append(p_chrome)
time.sleep(15)

# 5. Simulate interactions with xdotool
def type_text(text):
    for ch in text:
        if ch == ' ':
            subprocess.run(['xdotool', 'key', 'space'], capture_output=True)
        elif ch == '\n':
            subprocess.run(['xdotool', 'key', 'Return'], capture_output=True)
        elif ch.isupper() or ch in '!@#$%^&*()_+{}:"|<>?~':
            subprocess.run(['xdotool', 'key', f'shift+{ch.lower()}'], capture_output=True)
        else:
            subprocess.run(['xdotool', 'key', ch], capture_output=True)
        time.sleep(0.02)

def click(x, y):
    subprocess.run(['xdotool', 'mousemove', str(x), str(y), 'click', '1'], capture_output=True)

def screenshot(name):
    subprocess.run(['import', '-window', 'root', name], capture_output=True, timeout=10)

print("Starting interactions...")

# Screenshot 1: Initial page load
time.sleep(5)
screenshot(os.path.join(BASE_DIR, 'demo_step1_landing.png'))

# Click Generate button (center-left area, around x=200, y=580)
# In Streamlit with 1280x720, 2-column layout, the button is in left column
click(200, 520)
print("Clicked Generate...")
time.sleep(90)  # Wait for generation (models + inference)

# Screenshot 2: After generation
time.sleep(5)
screenshot(os.path.join(BASE_DIR, 'demo_step2_generated.png'))

# Switch to Inpainting tab (top of page, around x=400, y=100-120)
click(500, 110)
time.sleep(3)

# Screenshot 3: Inpainting tab
time.sleep(3)
screenshot(os.path.join(BASE_DIR, 'demo_step3_inpainting.png'))

# Draw on canvas - draw a line/dot on the canvas
# Canvas is around left half, middle
subprocess.run(['xdotool', 'mousemove', '250', '400'], capture_output=True)
time.sleep(0.5)
subprocess.run(['xdotool', 'mousedown', '1'], capture_output=True)
time.sleep(0.1)
subprocess.run(['xdotool', 'mousemove', '350', '450'], capture_output=True)
subprocess.run(['xdotool', 'mousemove', '280', '350'], capture_output=True)
time.sleep(0.3)
subprocess.run(['xdotool', 'mouseup', '1'], capture_output=True)
time.sleep(1)

# Click Run Inpainting button
click(200, 680)
print("Clicked Inpainting...")
time.sleep(90)  # Wait for inpainting

# Screenshot 4: Inpainting result
time.sleep(5)
screenshot(os.path.join(BASE_DIR, 'demo_step4_inpainted.png'))

# Scroll down to Zoom-Out
subprocess.run(['xdotool', 'click', '4'])  # scroll up to see top
time.sleep(1)
subprocess.run(['xdotool', 'click', '5'])  # scroll down
time.sleep(1)

# Click Run Zoom-Out button
click(200, 650)
print("Clicked Zoom-Out...")
time.sleep(120)  # Wait for zoom-out

# Screenshot 5: Final result
time.sleep(5)
screenshot(os.path.join(BASE_DIR, 'demo_step5_zoom.png'))

print("Recording complete, cleaning up...")
cleanup()
disp.stop()
print(f"Video saved to {output_path}")
