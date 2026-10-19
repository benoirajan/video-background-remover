#!/usr/bin/env python3
"""
Professional Video Matting Pipeline using Robust Video Matting (RVM).
Generates:
1. Alpha Matte Video (black and white continuous alpha for Kdenlive masking)
2. Green-Screen / Custom background composite with edge despill
3. Side-by-side comparison video
"""

import os
import sys
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import subprocess
from tqdm import tqdm

import shutil

def run_ffmpeg(args_list):
    """Run ffmpeg using system ffmpeg, or snap kdenlive ffmpeg, or return False."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg"] + args_list
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    # Try through snap run --shell kdenlive
    try:
        quoted = " ".join([f'"{a}"' if ' ' in a or ';' in a else a for a in args_list])
        cmd = ["snap", "run", "--shell", "kdenlive", "-c", f"ffmpeg {quoted}"]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False

def ensure_model_exists(model_path):
    """Auto-download the ONNX model if not found locally."""
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    filename = os.path.basename(model_path)
    if filename in ["rvm_resnet50_fp32.onnx", "rvm_mobilenetv3_fp32.onnx"]:
        url = f"https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/{filename}"
        print(f"\nModel not found locally: {filename}")
        print(f"Downloading from {url} ...")
        import urllib.request
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100.0, (downloaded / total_size) * 100)
                sys.stdout.write(f"\rDownloading model: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, model_path, reporthook=progress)
        print("\nModel download finished successfully.\n")
    return model_path

def process_video(
    input_path,
    output_dir,
    model_path,
    downsample_ratio=0.25,
    alpha_threshold=0.03,
    max_frames=None,
    start_frame=0,
    bg_mode="green", # 'green', 'black', 'white', or RGB tuple
    generate_matte=True,
    generate_composite=True,
    generate_preview=True
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    model_path = ensure_model_exists(model_path)
    print(f"Loading RVM ONNX Model from: {model_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    
    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    print(f"Execution Provider: {providers[0]}")
    
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = max(0, total_frames - start_frame)
        
    if max_frames and max_frames < total_frames:
        total_frames = max_frames
        
    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height} @ {fps:.2f} fps ({total_frames} frames to process)")
    
    # Recurrent states for RVM
    r1i = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r2i = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r3i = np.zeros((1, 1, 1, 1), dtype=np.float32)
    r4i = np.zeros((1, 1, 1, 1), dtype=np.float32)
    dsr = np.array([downsample_ratio], dtype=np.float32)
    
    # Setup background color
    if bg_mode == "green":
        bg_color = np.array([0, 255, 0], dtype=np.float32) # BGR
    elif bg_mode == "black":
        bg_color = np.array([0, 0, 0], dtype=np.float32)
    elif bg_mode == "white":
        bg_color = np.array([255, 255, 255], dtype=np.float32)
    else:
        bg_color = np.array([0, 255, 0], dtype=np.float32)
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writers = {}
    matte_path = os.path.join(output_dir, f"{base_name}_alpha_matte_temp.mp4")
    comp_path = os.path.join(output_dir, f"{base_name}_composite_temp.mp4")
    prev_path = os.path.join(output_dir, f"{base_name}_preview_temp.mp4")
    
    if generate_matte:
        writers['matte'] = cv2.VideoWriter(matte_path, fourcc, fps, (width, height), isColor=False)
    if generate_composite:
        writers['comp'] = cv2.VideoWriter(comp_path, fourcc, fps, (width, height), isColor=True)
    if generate_preview:
        # Side by side: (width * 2, height)
        writers['prev'] = cv2.VideoWriter(prev_path, fourcc, fps, (width * 2, height), isColor=True)
        
    print("Processing frames with temporal video matting...")
    processed_count = 0
    pbar = tqdm(total=total_frames, desc="Matting Progress")
    
    try:
        while cap.isOpened() and (max_frames is None or processed_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # OpenCV BGR -> RGB float32 [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            src = np.transpose(frame_rgb, (2, 0, 1))[np.newaxis, ...] # [1, 3, H, W]
            
            # Inference
            fgr, pha, r1o, r2o, r3o, r4o = session.run(None, {
                'src': src,
                'r1i': r1i, 'r2i': r2i, 'r3i': r3i, 'r4i': r4i,
                'downsample_ratio': dsr
            })
            
            # Update recurrent state
            r1i, r2i, r3i, r4i = r1o, r2o, r3o, r4o
            
            # Alpha matte [H, W] in [0, 1]
            alpha = pha[0, 0]
            if alpha_threshold > 0:
                # Suppress background noise / shadows below threshold
                alpha = np.where(alpha < alpha_threshold, 0.0, alpha)
                # Smoothly normalize remainder back to [0, 1]
                alpha = np.clip((alpha - alpha_threshold) / (1.0 - alpha_threshold + 1e-7), 0.0, 1.0)
            alpha_uint8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            
            # Foreground [H, W, 3] in [0, 1]
            fg = np.transpose(fgr[0], (1, 2, 0)) # RGB
            fg_bgr = cv2.cvtColor(np.clip(fg * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Continuous alpha blending
            alpha_3d = alpha[:, :, np.newaxis]
            comp_bgr = (fg_bgr.astype(np.float32) * alpha_3d + bg_color * (1.0 - alpha_3d)).astype(np.uint8)
            
            if generate_matte:
                writers['matte'].write(alpha_uint8)
            if generate_composite:
                writers['comp'].write(comp_bgr)
            if generate_preview:
                sbs = np.hstack([frame, comp_bgr])
                writers['prev'].write(sbs)
                
            processed_count += 1
            pbar.update(1)
            
    finally:
        pbar.close()
        cap.release()
        for w in writers.values():
            w.release()
            
    print(f"Finished {processed_count} frames.")
    
    print("Finalizing video files...")
    
    final_matte = os.path.join(output_dir, f"{base_name}_alpha_matte.mp4")
    final_comp = os.path.join(output_dir, f"{base_name}_composite.mp4")
    final_prev = os.path.join(output_dir, f"{base_name}_preview.mp4")
    
    def mux_video(temp_file, final_file, with_audio=True):
        if not os.path.exists(temp_file):
            return
        if with_audio:
            args = [
                "-y",
                "-i", temp_file,
                "-ss", f"{start_frame/fps}",
                "-i", input_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                final_file
            ]
        else:
            args = [
                "-y",
                "-i", temp_file,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                final_file
            ]
        success = run_ffmpeg(args)
        if success and os.path.exists(final_file) and os.path.getsize(final_file) > 0:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        else:
            if os.path.exists(temp_file):
                os.replace(temp_file, final_file)
            
    if generate_matte:
        mux_video(matte_path, final_matte, with_audio=False)
        print(f"Created Alpha Matte: {final_matte}")
    if generate_composite:
        mux_video(comp_path, final_comp, with_audio=True)
        print(f"Created Composite Video: {final_comp}")
    if generate_preview:
        mux_video(prev_path, final_prev, with_audio=True)
        print(f"Created Preview Video: {final_prev}")
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Video Matting with RVM")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--output-dir", default="/home/benoi/Videos/matting_project/output", help="Output directory")
    parser.add_argument("--model", default="resnet50", help="Model choice: 'resnet50' (highest quality), 'mobilenetv3' (faster), or path to .onnx file")
    parser.add_argument("--downsample-ratio", type=float, default=0.25, help="Downsample ratio (0.25 for 1080p landscape, 0.35-0.4 for portrait or fine details)")
    parser.add_argument("--alpha-threshold", type=float, default=0.03, help="Background noise suppression threshold (default: 0.03)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (useful for quick test)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--bg", default="green", choices=["green", "black", "white"], help="Background color for composite")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model
    if args.model.lower() in ("resnet50", "resnet", "r50"):
        model_path = os.path.join(script_dir, "models", "rvm_resnet50_fp32.onnx")
    elif args.model.lower() in ("mobilenetv3", "mobilenet", "mbv3"):
        model_path = os.path.join(script_dir, "models", "rvm_mobilenetv3_fp32.onnx")
        
    process_video(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=model_path,
        downsample_ratio=args.downsample_ratio,
        alpha_threshold=args.alpha_threshold,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        bg_mode=args.bg
    )
