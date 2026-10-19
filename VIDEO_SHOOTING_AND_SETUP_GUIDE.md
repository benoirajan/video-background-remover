# 🎬 AI Video Matting: Shooting Best Practices & Setup Guide

A complete guide for achieving studio-grade background removal without a green screen, optimizing video capture, and deploying the automated pipeline across local machines and cloud GPUs.

---

## 📌 Table of Contents
1. [How AI Video Matting Works](#1-how-ai-video-matting-works)
2. [Video Shooting Best Practices (Camera & Lighting)](#2-video-shooting-best-practices)
3. [Plug-and-Play Setup Guide (Local & Cloud)](#3-plug-and-play-setup-guide)
4. [Cloud Deployment Options (Colab, RunPod, AWS/GCP, Docker)](#4-cloud-deployment-options)
5. [Pipeline Configuration & Fine-Tuning Guide](#5-pipeline-configuration--fine-tuning-guide)
6. [Kdenlive & Video Editing Integration](#6-kdenlive--video-editing-integration)
7. [Troubleshooting Common Issues](#7-troubleshooting-common-issues)

---

## 1. How AI Video Matting Works

This pipeline uses **Robust Video Matting (RVM)**. Unlike traditional green-screen chroma keying (which relies strictly on a single color channel) or single-frame AI segmenters (which flicker violently between frames), RVM is a **recurrent neural network**:
* **Spatial Feature Extraction:** Identifies the human silhouette and separates foreground colors from background colors.
* **Temporal Recurrent Memory ($r_1, r_2, r_3, r_4$):** Remembers the position of the subject and background over consecutive frames, creating smooth, flicker-free boundaries.
* **Continuous Alpha Output:** Generates true 8-bit alpha transparency ($0$ to $255$), preserving fine sub-pixel elements such as individual hairs, translucent glasses, and semi-transparent edges.

---

## 2. Video Shooting Best Practices

The quality of your AI cutout is 80% determined by how the video is shot. Follow these practical rules to get crisp, professional results without green screens:

### A. Distance & Shadow Elimination (The 2-Meter Rule)
* **Never stand flat against a wall:** Maintain at least **1.5 to 2 meters (5 to 6 feet)** of distance between the subject and the background.
* **Why:** When you stand close to a wall, your body casts a hard, dark shadow. The AI cannot tell where your shirt ends and your shadow begins, resulting in unwanted dark patches sticking to your silhouette.

### B. Lighting: The Secret to Crisp Edges
* **High Ambient / Key Light:** Ensure your face and body are well-lit. Dim indoor lighting causes phone cameras to drop shutter speeds, causing motion blur.
* **Rim Light / Backlight (The Pro Tip):** Place a small light (a desk lamp, LED stick, or ceiling spot) behind you or overhead, aimed down at your shoulders and hair. This creates a subtle highlight around your perimeter that physically separates you from the background.
* **Even Background Illumination:** Avoid having strong direct spotlights aimed at the wall behind you that create harsh hot spots.

### C. Wardrobe & Clothing Contrast
* **Color Contrast:** Choose clothing colors that contrast strongly with your wall or room environment:
  * If your wall is white/beige: Wear dark or saturated colors (navy, black, maroon, forest green).
  * If your wall is dark wood/brick: Wear lighter or brighter shirts.
* **Avoid Sheer / Translucent Fabrics:** Extremely thin mesh or see-through lace confuses alpha estimation.
* **Watch Out for Similar Tones:** Avoid wearing a beige top against a beige wall, or a black sweater against a dark leather chair.

### D. Camera Setup & Motion Blur
* **Use a Stationary Tripod:** RVM relies on temporal memory across frames. If the camera is handheld and constantly wobbling, the temporal recurrent state becomes noisy. Keep the camera stationary.
* **Avoid Fast Motion Blur:** Fast hand waving or sudden head jerks in low light smear across frames. If you gesture with your hands, move naturally and ensure lighting is bright enough so the camera maintains a fast shutter speed (ideally $1/60\text{s}$ or faster).
* **Lock Auto-Exposure & Auto-Focus (AE/AF Lock):** On your smartphone, tap and hold on your face to lock focus and exposure so the camera does not pulsate brightness or hunt for focus mid-sentence.

---

## 3. Plug-and-Play Setup Guide

The repository includes a self-configuring setup system that automatically handles dependencies, virtual environments, and model weights.

### Quick Start (After cloning the repository)
```bash
git clone <your-repository-url>
cd matting_project

# 1. Run the universal installer
./setup.sh

# 2. Process your video
./run.sh path/to/your_video.mp4
```

### What `setup.sh` Automates:
1. **GPU vs. CPU Auto-Detection:** Checks for NVIDIA drivers via `nvidia-smi`.
   * If an **NVIDIA GPU** is present: installs `onnxruntime-gpu` with CUDA acceleration.
   * If **CPU-only**: installs standard multi-threaded `onnxruntime`.
2. **System Dependencies:** Checks for `ffmpeg` (required for audio muxing).
3. **Model Auto-Download:** Automatically downloads the high-accuracy `rvm_resnet50_fp32.onnx` model (~103 MB) from official releases.
4. **Environment Isolation:** Provisions an isolated `venv/` directory.

---

## 4. Cloud Deployment Options

Running on a cloud GPU accelerates processing from **~1 fps (CPU)** to **40–80+ fps (GPU)**—a **30× to 50× speedup**.

### Option A: Free Google Colab (Easiest)
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
3. In a notebook cell, run:
   ```bash
   !git clone <your-repo-url> matting_project
   %cd matting_project
   !./setup.sh --gpu
   !./run.sh /content/your_video.mp4 --downsample-ratio 0.35
   ```

### Option B: Cloud GPU VMs (RunPod, Lambda Labs, AWS, GCP)
1. Launch any Ubuntu VM with an NVIDIA GPU (RTX 3090, RTX 4090, or A10G).
2. SSH into your instance:
   ```bash
   git clone <your-repo-url>
   cd matting_project
   ./setup.sh
   ./run.sh input_video.mp4 --downsample-ratio 0.35
   ```

### Option C: Docker Deployment
A production-ready [`Dockerfile`](file:///home/benoi/Videos/matting_project/Dockerfile) is provided:
```bash
# Build the GPU container
docker build -t video-matting:latest .

# Run container with NVIDIA GPU passthrough
docker run --gpus all --rm \
  -v $(pwd)/input.mp4:/workspace/input.mp4 \
  -v $(pwd)/output:/workspace/output \
  video-matting:latest /workspace/input.mp4 --output-dir /workspace/output --downsample-ratio 0.35
```

---

## 5. Pipeline Configuration & Fine-Tuning Guide

Command options available in `run.sh` / `process_video.py`:

```bash
./run.sh <input_video.mp4> [OPTIONS]
```

| Parameter | Recommended Setting | Description |
| :--- | :--- | :--- |
| `--model` | `resnet50` (Default) | Choose `resnet50` for maximum accuracy, or `mobilenetv3` for high-speed mobile/lightweight inference. |
| `--downsample-ratio` | `0.35` to `0.45` | Scale of the internal feature map. Use `0.25` for 1080p landscape, but **`0.35` to `0.4` for vertical 1080×1920 portrait videos** to keep fingers, hair, and edges sharp. |
| `--alpha-threshold` | `0.03` to `0.05` | Noise floor cutoff. Clamps faint 1%–5% background shadows/reflections directly to pure black ($0.0$). |
| `--max-frames` | e.g. `90` | Limits execution to the first $N$ frames (useful for 3-second quick testing before processing a long file). |
| `--start-frame` | e.g. `300` | Skips initial frames to start processing from frame index $N$. |
| `--bg` | `green` (Default) | Composite background color: `green`, `black`, or `white`. |

### Recommended Commands by Use Case:

* **High-Quality Portrait / Shorts / Reels Video (1080×1920):**
  ```bash
  ./run.sh my_short.mp4 --downsample-ratio 0.35 --alpha-threshold 0.04
  ```
* **Quick 5-Second Test Clip:**
  ```bash
  ./run.sh my_video.mp4 --max-frames 150
  ```
* **Clean Alpha Matte with Black Background:**
  ```bash
  ./run.sh my_video.mp4 --bg black
  ```

---

## 6. Kdenlive & Video Editing Integration

Each run generates three outputs in `output/`:
1. `<name>_composite.mp4`: Subject composited on clean green with audio preserved.
2. `<name>_alpha_matte.mp4`: High-contrast grayscale mask (White = Subject, Black = Transparent).
3. `<name>_preview.mp4`: Side-by-side comparison of original vs. cutout.

### Workflow A: Chroma Keying the Composite (Fast & Clean)
1. In Kdenlive, place your **New Background** on **Video Track 1 (V1)**.
2. Place `<name>_composite.mp4` on **Video Track 2 (V2)**.
3. Add the **Chroma Key: Basic** effect to Track 2.
4. Use the **Color Picker Pipette** to sample the green background.
5. Set the **Variance** slider between `25` and `35`. Because the composite was rendered with anti-aliased alpha blending, edges remain clean without green spill.

### Workflow B: True Luma / Alpha Matte Masking (Studio VFX Quality)
1. Setup 3 Video Tracks in Kdenlive:
   * **Track 3 (Top):** `<name>_alpha_matte.mp4`
   * **Track 2 (Middle):** Original raw camera video (`VID_...mp4`)
   * **Track 1 (Bottom):** New background
2. Add a **Composite & Transform** or **Mask Apply** effect from Track 2 to Track 3 using the **Alpha / Luma channel**.
3. **Advantage:** Preserves 100% of the camera's original color grade and dynamic range with zero chroma degradation.

---

## 7. Troubleshooting Common Issues

### Issue 1: "Small patches of background flicker or stay visible"
* **Fix:** Increase the noise filter: `--alpha-threshold 0.05` or `0.06`.
* **Filming Fix:** Move the subject 1 meter further away from the wall to drop background shadows.

### Issue 2: "Fingers or edges of clothing disappear or get eaten away"
* **Fix:** Increase the internal resolution: `--downsample-ratio 0.4` or `0.45`.
* **Fix:** Ensure the ResNet50 model is selected: `--model resnet50`.

### Issue 3: "Fast hand movements look blurry or cut off"
* **Filming Fix:** Increase lighting on the subject so your camera records with a shutter speed of at least $1/60\text{s}$ (eliminating motion blur streaks).

### Issue 4: "Processing is slow on laptop CPU"
* **Fix:** For local laptop testing, use `--max-frames 90` to preview a 3-second sample. For full renders, run on a free Google Colab T4 GPU or cloud instance.
