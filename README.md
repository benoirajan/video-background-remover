# AI Video Matting Pipeline & Kdenlive Integration Guide

This project provides an automated, professional background removal and matting pipeline for video footage recorded **without a green screen**. It uses **Robust Video Matting (RVM)** via an ONNX runtime engine optimized for your Intel CPU.

---

## 📁 Project Structure

```text
/home/benoi/Videos/matting_project/
├── models/
│   └── rvm_mobilenetv3_fp32.onnx   # Pre-trained RVM neural network (~15 MB)
├── venv/                           # Isolated Python environment (ONNX + OpenCV)
├── output/                         # Processed video outputs & previews
├── process_video.py                # Main matting & video muxing script
├── run.sh                          # Easy executable runner script
└── README.md                       # This documentation
```

---

## 🚀 How to Run the Matting Pipeline

### 1. Simple Command (Using `run.sh`)

Open your terminal, navigate to the folder, and run:

```bash
cd /home/benoi/Videos/matting_project
./run.sh /home/benoi/Videos/VID_20261017_210103472.mp4
```

Or using the full Python virtual environment path directly from anywhere:

```bash
/home/benoi/Videos/matting_project/venv/bin/python3 /home/benoi/Videos/matting_project/process_video.py /home/benoi/Videos/VID_20261017_210103472.mp4
```

---

### 2. Common Options & Flags

| Flag | Purpose | Example |
| :--- | :--- | :--- |
| `--model <name>` | Select backbone model: `resnet50` (highest accuracy, default) or `mobilenetv3` (faster) | `./run.sh input.mp4 --model resnet50` |
| `--downsample-ratio <R>` | Downsample scale for feature extraction (`0.25` for 1080p landscape, `0.35`–`0.45` for vertical/portrait or fine details) | `./run.sh input.mp4 --downsample-ratio 0.35` |
| `--alpha-threshold <T>` | Suppress faint background haze/shadows (Default: `0.03`, range: `0.01`–`0.08`) | `./run.sh input.mp4 --alpha-threshold 0.05` |
| `--max-frames <N>` | Process only the first $N$ frames (ideal for quick tests) | `./run.sh input.mp4 --max-frames 90` |
| `--start-frame <N>` | Start processing from frame index $N$ | `./run.sh input.mp4 --start-frame 300` |
| `--bg <color>` | Change background color for composite (`green`, `black`, `white`) | `./run.sh input.mp4 --bg black` |
| `--output-dir <path>` | Specify a custom output directory | `./run.sh input.mp4 --output-dir ./output` |

---

## 📦 What the Pipeline Generates

For every processed video (`<name>.mp4`), three files are saved in `/home/benoi/Videos/matting_project/output/`:

1. **`<name>_composite.mp4`**: The subject cleanly cut out and composited on pure green (with original audio preserved).
2. **`<name>_alpha_matte.mp4`**: A high-contrast continuous grayscale video (White = subject, Black = background, Gray = fine hair/edges).
3. **`<name>_preview.mp4`**: A side-by-side comparison video showing the raw footage vs. the cutout.

---

## 🎬 How to Use the Output in Kdenlive

You have two methods to use these files inside Kdenlive:

### Method A: Chroma Keying the Composite (Recommended – Fast & Easy)

Because `<name>_composite.mp4` was blended using continuous mathematical alpha values, chroma keying it in Kdenlive produces crisp, clean edges without green fringing or jitter.

1. **Launch Kdenlive**: Open Kdenlive from your application menu or run `kdenlive` in the terminal.
2. **Add Tracks**: Ensure you have at least two video tracks in your timeline:
   * **Video Track 2 (V2)** (Top track)
   * **Video Track 1 (V1)** (Bottom track)
3. **Place Media**:
   * Drag your **New Background** (image, graphic, or scenery video) onto **Track 1 (V1)**.
   * Drag the generated **`<name>_composite.mp4`** onto **Track 2 (V2)** directly above the background.
4. **Apply Chroma Key**:
   * Select the clip on **V2**.
   * Go to the **Effects** tab (or menu: *View $\rightarrow$ Effects*).
   * Search for **Chroma Key: Basic** and drag it onto the clip on **V2**.
   * Click the **Color Picker (Pipette)** tool and click the green background in the preview monitor.
   * Adjust the **Variance** slider slightly (usually between `20` and `45`) until the green disappears cleanly.

---

### Method B: True Alpha Matte Masking (Studio VFX Technique)

This technique uses your **original raw video** at 100% original color fidelity, while the **`<name>_alpha_matte.mp4`** controls the transparency.

1. **Timeline Setup (3 Tracks)**:
   * **Video Track 3 (V3)**: `<name>_alpha_matte.mp4`
   * **Video Track 2 (V2)**: Original raw video (`VID_...mp4`)
   * **Video Track 1 (V1)**: New background
2. **Apply the Alpha Mask**:
   * Right-click the clip on **Track 2** or **Track 3** and add a **Composition / Transition**:
     * Choose **Composite and Transform** or **Wipe (Matte)**.
     * In the transition settings, set the transparency source to **Luma / Alpha** from Track 3.
   * Alternatively, add the **Mask Apply** effect to Track 2 referencing Track 3.

---

## 🎨 Pro Tips for a Professional Finish in Kdenlive

1. **Subtle Edge Feathering**:
   * If you notice any hard edges, add a mild **Edge Blur** (radius of 1 to 2 pixels) to the composite or mask. This creates a natural lens-blur transition.
2. **Light Wrap & Color Matching**:
   * In reality, light from the background wraps around a subject.
   * Add a **Color Correction** or **Lift/Gamma/Gain** effect to your subject clip (Track 2) to match the ambient warmth/coolness and brightness of your new background on Track 1.
