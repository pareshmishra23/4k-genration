#!/usr/bin/env python3
"""
4K Video Upscaler - CPU/GPU compatible
Adapted from: https://github.com/yuvraj108c/4k-video-upscaler-colab
Uses Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN

Models:
  - RealESRGAN_x4plus
  - RealESRGAN_x4plus_anime_6B
  - realesr-animevideov3

Usage:
  python upscale_video.py --input video.mp4 --output /output --resolution "4k (3840 x 2160)" --model RealESRGAN_x4plus
"""

import argparse
import os
import subprocess
import cv2
import torch
import sys

# Resolve Real-ESRGAN paths
REAL_ESRGAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Real-ESRGAN")
if os.path.isdir(REAL_ESRGAN_DIR):
    sys.path.insert(0, REAL_ESRGAN_DIR)

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np


def get_upsampler(model_name: str, tile: int = 0, device: str = "cpu"):
    """Create a RealESRGANer upsampler for the given model."""
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
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    elif model_name == "realesr-animevideov3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Download weights
    from basicsr.utils.download_util import load_file_from_url

    if isinstance(file_url, list):
        model_paths = []
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(REAL_ESRGAN_DIR, "weights"), progress=True, file_name=None
            )
            model_paths.append(model_path)
        model_path = model_paths
        dni_weight = [0.5, 0.5]
    else:
        model_path = load_file_from_url(
            url=file_url, model_dir=os.path.join(REAL_ESRGAN_DIR, "weights"), progress=True, file_name=None
        )
        dni_weight = None

    # Determine device and half precision
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


def upscale_video(
    video_path: str,
    output_dir: str,
    resolution: str,
    model: str,
    device: str = "cpu",
    tile: int = 0,
):
    """Upscale a video to the target resolution."""
    assert os.path.exists(video_path), f"Video file does not exist: {video_path}"

    video_capture = cv2.VideoCapture(video_path)
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.release()

    final_width = None
    final_height = None
    aspect_ratio = float(video_width / video_height)

    # Parse resolution options
    resolution_map = {
        "FHD (1920 x 1080)": (1920, 1080),
        "2k (2560 x 1440)": (2560, 1440),
        "4k (3840 x 2160)": (3840, 2160),
    }

    if resolution in resolution_map:
        final_width, final_height = resolution_map[resolution]
    elif "x original" in resolution:
        multiplier = int(resolution.split("x")[0].strip())
        final_width = multiplier * video_width
        final_height = multiplier * video_height
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    # Adjust for aspect ratio
    if aspect_ratio == 1.0 and "original" not in resolution:
        final_height = final_width

    if aspect_ratio < 1.0 and "original" not in resolution:
        final_width, final_height = final_height, final_width

    scale_factor = max(final_width / video_width, final_height / video_height)

    # Ensure even dimensions
    while True:
        scaled_w = int(video_width * scale_factor)
        scaled_h = int(video_height * scale_factor)
        if scaled_w % 2 == 0 and scaled_h % 2 == 0:
            break
        scale_factor += 0.01

    print(f"Input: {video_width}x{video_height}")
    print(f"Target: {final_width}x{final_height}")
    print(f"Scale factor: {scale_factor:.2f}")
    print(f"Model: {model}")
    print(f"Device: {device}")

    # Create upsampler
    upsampler = get_upsampler(model, tile=tile, device=device)

    # Process video frame by frame
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    upscaled_video_path = os.path.join(output_dir, f"{video_name}_out.mp4")

    # Calculate output dimensions
    out_width = int(video_width * scale_factor)
    out_height = int(video_height * scale_factor)

    # Open output video writer (CPU-safe, no NVENC)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        upscaled_video_path,
        fourcc,
        fps,
        (out_width, out_height),
    )

    frame_idx = 0
    from tqdm import tqdm

    pbar = tqdm(total=total_frames, desc="Upscaling")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        try:
            output, _ = upsampler.enhance(frame, outscale=scale_factor)
            out_writer.write(output)
        except RuntimeError as error:
            print(f"Error on frame {frame_idx}: {error}")
            print("Try reducing --tile size or using a smaller model.")
            # Write original frame as fallback
            resized = cv2.resize(frame, (out_width, out_height))
            out_writer.write(resized)

        frame_idx += 1
        pbar.update(1)

    video_capture.release()
    out_writer.release()
    pbar.close()

    print(f"Upscaled video saved to: {upscaled_video_path}")

    # Crop to exact final resolution if needed
    final_video_path = os.path.join(output_dir, f"{video_name}_upscaled_{final_width}_{final_height}.mp4")
    if "original" not in resolution:
        print("Cropping to fit target resolution...")
        cmd = [
            "ffmpeg", "-y",
            "-i", upscaled_video_path,
            "-filter:v", f"crop={final_width}:{final_height}:(in_w-{final_width})/2:(in_h-{final_height})/2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            final_video_path,
        ]
        subprocess.run(cmd, check=True)
    else:
        os.replace(upscaled_video_path, final_video_path)

    # Cleanup intermediate
    if os.path.exists(upscaled_video_path):
        os.remove(upscaled_video_path)

    print(f"Final upscaled video: {final_video_path}")
    return final_video_path


def main():
    parser = argparse.ArgumentParser(description="4K Video Upscaler using Real-ESRGAN")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory")
    parser.add_argument(
        "--resolution",
        type=str,
        default="4k (3840 x 2160)",
        choices=[
            "FHD (1920 x 1080)",
            "2k (2560 x 1440)",
            "4k (3840 x 2160)",
            "2 x original",
            "3 x original",
            "4 x original",
        ],
        help="Target resolution",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RealESRGAN_x4plus",
        choices=[
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRNet_x4plus",
            "RealESRGAN_x2plus",
            "realesr-animevideov3",
            "realesr-general-x4v3",
        ],
        help="Model to use",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto' (default)")
    parser.add_argument("--tile", type=int, default=0, help="Tile size for processing (0=auto, smaller for less VRAM)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage (alias for --device cuda)")
    args = parser.parse_args()

    # Determine device
    if args.gpu:
        device = "cuda"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    upscaled = upscale_video(
        video_path=args.input,
        output_dir=args.output,
        resolution=args.resolution,
        model=args.model,
        device=device,
        tile=args.tile,
    )
    print(f"\nDone! Output: {upscaled}")


if __name__ == "__main__":
    main()
