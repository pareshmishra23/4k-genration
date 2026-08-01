#!/usr/bin/env python3
"""
4K Video Upscaler - CPU/GPU compatible
Adapted from: https://github.com/yuvraj108c/4k-video-upscaler-colab
Uses Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN

Models:
  - RealESRGAN_x4plus
  - RealESRGAN_x4plus_anime_6B
  - realesr-animevideov3
  - RealESRNet_x4plus
  - RealESRGAN_x2plus
  - realesr-general-x4v3

Usage:
  python upscale_video.py --input video.mp4 --output /output --resolution "4k (3840 x 2160)" --model RealESRGAN_x4plus
"""

import argparse
import os
import subprocess
import cv2
import torch
import sys
import pathlib
import importlib.util

def patch_basicsr():
    """Patch basicsr to fix torchvision compatibility issue and numpy 2.0 issues."""
    try:
        # Check for basicsr module
        spec = importlib.util.find_spec("basicsr")
        if spec is None:
            print("basicsr not found, skipping patch.")
            return

        import basicsr
        basicsr_path = pathlib.Path(basicsr.__file__).parent

        # Patch torchvision.transforms.functional_tensor
        degradations_path = basicsr_path / "data" / "degradations.py"
        if degradations_path.exists():
            content = degradations_path.read_text()
            if "functional_tensor" in content:
                print("Patching basicsr for torchvision compatibility...")
                content = content.replace(
                    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                    "from torchvision.transforms.functional import rgb_to_grayscale",
                )
                try:
                    degradations_path.write_text(content)
                    print("Successfully patched basicsr torchvision import.")
                except PermissionError:
                    print("Warning: Could not patch basicsr torchvision import due to permission error.")

        # Patch numpy 2.0 compatibility if needed (e.g., np.int is deprecated)
        # This is a more general approach, specific files might need targeted patches
        # For now, we rely on pinning numpy<2 in requirements.txt

    except Exception as e:
        print(f"Note: Basicsr patch skipped or failed: {e}")

# Run patch before imports that might fail
patch_basicsr()

# Resolve Real-ESRGAN paths
REAL_ESRGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Real-ESRGAN")
if os.path.isdir(REAL_ESRGAN_DIR):
    sys.path.insert(0, REAL_ESRGAN_DIR)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.utils.download_util import load_file_from_url
except ImportError:
    print("Error: Required libraries (basicsr, realesrgan) not found. Please install them first.")
    sys.exit(1)

import numpy as np
from tqdm import tqdm

def get_upsampler(model_name: str, tile: int = 0, device: str = "cpu"):
    """Create a RealESRGANer upsampler for the given model."""
    weights_dir = os.path.join(REAL_ESRGAN_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "realesr-animevideov3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if isinstance(file_url, list):
        model_paths = []
        for url in file_url:
            model_path = load_file_from_url(url=url, model_dir=weights_dir, progress=True, file_name=None)
            model_paths.append(model_path)
        model_path = model_paths
        dni_weight = [0.5, 0.5]
    else:
        model_path = load_file_from_url(url=file_url, model_dir=weights_dir, progress=True, file_name=None)
        dni_weight = None

    use_half = device == "cuda"
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=torch.device(device),
    )
    return upsampler

def upscale_video(video_path, output_dir, resolution, model_name, device="cpu", tile=0):
    """Upscale a video to the target resolution."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    res_map = {
        "FHD (1920 x 1080)": (1920, 1080),
        "2k (2560 x 1440)": (2560, 1440),
        "4k (3840 x 2160)": (3840, 2160),
    }

    if resolution in res_map:
        target_w, target_h = res_map[resolution]
    elif "x original" in resolution:
        mult = int(resolution.split("x")[0].strip())
        target_w, target_h = width * mult, height * mult
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

    # Maintain aspect ratio logic
    aspect = width / height
    # Calculate the scaling factor based on the target resolution
    scale_factor_w = target_w / width
    scale_factor_h = target_h / height
    scale = min(scale_factor_w, scale_factor_h) # Use min to ensure the video fits within the target dimensions

    out_w, out_h = int(width * scale), int(height * scale)
    # Ensure even dimensions for ffmpeg
    if out_w % 2 != 0: out_w -= 1 # Subtract 1 to make it even
    if out_h % 2 != 0: out_h -= 1 # Subtract 1 to make it even

    print(f"Processing: {width}x{height} -> {out_w}x{out_h} (Target: {target_w}x{target_h})")
    
    upsampler = get_upsampler(model_name, tile=tile, device=device)
    
    os.makedirs(output_dir, exist_ok=True)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    temp_output = os.path.join(output_dir, f"{video_base}_temp.mp4")
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))
    
    pbar = tqdm(total=total_frames, desc="Upscaling")
    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            # RealESRGANer expects RGB, OpenCV reads BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_rgb, _ = upsampler.enhance(frame_rgb, outscale=scale)
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            writer.write(output_bgr)
        except Exception as e:
            print(f"Error: {e}. Resizing as fallback.")
            writer.write(cv2.resize(frame, (out_w, out_h)))
        pbar.update(1)
    
    cap.release()
    writer.release()
    pbar.close()

    # Final crop and encode with ffmpeg
    final_output = os.path.join(output_dir, f"{video_base}_upscaled_{target_w}x{target_h}.mp4")
    print("Finalizing video with ffmpeg...")
    
    # Ensure the crop filter uses the target_w and target_h for the final output dimensions
    # and centers the crop from the potentially slightly larger upscaled video
    crop_filter = f"crop={target_w}:{target_h}:(iw-{target_w})/2:(ih-{target_h})/2"
    cmd = [
        "ffmpeg", "-y", "-i", temp_output,
        "-vf", crop_filter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        final_output
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        # If ffmpeg fails, just keep the temp_output as the final output
        os.rename(temp_output, final_output)
        print(f"FFmpeg failed, keeping intermediate file as final: {final_output}")
        return final_output

    os.remove(temp_output)
    
    print(f"Done! Saved to: {final_output}")
    return final_output

def main():
    parser = argparse.ArgumentParser(description="4K Video Upscaler")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--resolution", "-r", default="4k (3840 x 2160)", help="Target resolution")
    parser.add_argument("--model", "-m", default="RealESRGAN_x4plus", help="Model name")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu)")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0 for auto)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    upscale_video(args.input, args.output, args.resolution, args.model, device, args.tile)

if __name__ == "__main__":
    main()
