#!/usr/bin/env python3
"""
4K Video Upscaler - Enhanced Edition v2.0
==========================================
Based on:
  - https://github.com/yuvraj108c/4k-video-upscaler-colab
  - https://github.com/pareshmishra23/4k-genration

Key Improvements:
  1. Audio preservation (stream copy via FFmpeg)
  2. Lossless raw intermediate pipeline (no double compression)
  3. Smart auto-tiling based on VRAM / resolution
  4. Batch directory processing with resume support
  5. Proper BT.709 color space tagging
  6. Graceful per-frame error recovery with fallback resize
  7. Detailed progress with FPS and ETA
  8. Better memory management (raw video pipe, not cv2.VideoWriter)
  9. Downscale warning + validation
 10. Configurable CRF / preset for quality vs speed
 11. Web-optimized output (faststart moov atom)
 12. Dry-run mode for pipeline validation
"""

import argparse
import os
import subprocess
import sys
import pathlib
import importlib.util
import shutil
import json
import time
import tempfile
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Patch basicsr BEFORE any Real-ESRGAN imports
# ---------------------------------------------------------------------------
def patch_basicsr() -> bool:
    """Patch basicsr to fix torchvision>=0.17 compatibility issue."""
    try:
        spec = importlib.util.find_spec("basicsr")
        if spec is None:
            return False

        import basicsr
        basicsr_path = pathlib.Path(basicsr.__file__).parent
        degradations_path = basicsr_path / "data" / "degradations.py"

        if not degradations_path.exists():
            return False

        content = degradations_path.read_text()
        old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new_import = "from torchvision.transforms.functional import rgb_to_grayscale"

        if old_import in content:
            content = content.replace(old_import, new_import)
            try:
                degradations_path.write_text(content)
                print("[INFO] Patched basicsr for torchvision compatibility")
                return True
            except PermissionError:
                print("[WARN] Could not patch basicsr (permission denied)")
                return False
        return True
    except Exception as e:
        print(f"[WARN] Basicsr patch failed: {e}")
        return False


patch_basicsr()

# ---------------------------------------------------------------------------
# 2. Core imports
# ---------------------------------------------------------------------------
try:
    import cv2
    import torch
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    sys.exit(1)

# Resolve Real-ESRGAN path
SCRIPT_DIR = Path(__file__).parent.resolve()
# Handle Colab nested clone paths
for _ in range(5):
    if (SCRIPT_DIR / 'Real-ESRGAN').is_dir():
        break
    if SCRIPT_DIR.parent != SCRIPT_DIR:
        SCRIPT_DIR = SCRIPT_DIR.parent
    else:
        break
REAL_ESRGAN_DIR = SCRIPT_DIR / "Real-ESRGAN"
if REAL_ESRGAN_DIR.is_dir():
    sys.path.insert(0, str(REAL_ESRGAN_DIR))

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.utils.download_util import load_file_from_url
except ImportError as e:
    raise ImportError(
        f"Real-ESRGAN libraries not found: {e}. "
        "Please run: git clone https://github.com/xinntao/Real-ESRGAN.git && "
        "cd Real-ESRGAN && pip install ."
    ) from e


# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------
MODELS: Dict[str, Dict] = {
    "RealESRGAN_x4plus": {
        "class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64,
                   "num_block": 23, "num_grow_ch": 32, "scale": 4},
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "description": "Best for general photos and high-quality videos"
    },
    "RealESRNet_x4plus": {
        "class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64,
                   "num_block": 23, "num_grow_ch": 32, "scale": 4},
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "description": "General model with fewer artifacts"
    },
    "RealESRGAN_x4plus_anime_6B": {
        "class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64,
                   "num_block": 6, "num_grow_ch": 32, "scale": 4},
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "description": "Optimized for anime and cartoons"
    },
    "realesr-animevideov3": {
        "class": "SRVGGNetCompact",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64,
                   "num_conv": 16, "upscale": 4, "act_type": "prelu"},
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "description": "Fastest model for anime videos"
    },
    "RealESRGAN_x2plus": {
        "class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64,
                   "num_block": 23, "num_grow_ch": 32, "scale": 2},
        "netscale": 2,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "description": "Optimized for 2x upscaling"
    },
    "realesr-general-x4v3": {
        "class": "SRVGGNetCompact",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64,
                   "num_conv": 32, "upscale": 4, "act_type": "prelu"},
        "netscale": 4,
        "url": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        ],
        "description": "General model with noise reduction"
    }
}

RESOLUTION_MAP = {
    "FHD (1920 x 1080)": (1920, 1080),
    "2k (2560 x 1440)":   (2560, 1440),
    "4k (3840 x 2160)":   (3840, 2160),
    "8k (7680 x 4320)":   (7680, 4320),
}

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".wmv"}


# ---------------------------------------------------------------------------
# 4. Data structures
# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    pixel_format: str = "yuv420p"


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd: List[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with proper error handling."""
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0 and capture:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def get_video_info(video_path: str) -> VideoInfo:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration,pix_fmt,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    result = run_cmd(cmd)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fmt = data.get("format", {})

    # FPS
    fps_str = stream.get("r_frame_rate", "30/1")
    num, den = map(int, fps_str.split("/"))
    fps = num / den if den != 0 else 30.0

    # Frame count
    nb = stream.get("nb_frames")
    if nb and str(nb).isdigit():
        total_frames = int(nb)
    else:
        dur = float(fmt.get("duration", stream.get("duration", 0)) or 0)
        total_frames = int(dur * fps)

    # Audio check
    acmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name", "-of", "json", video_path]
    ares = subprocess.run(acmd, capture_output=True, text=True)
    has_audio = False
    audio_codec = None
    if ares.returncode == 0:
        adata = json.loads(ares.stdout)
        if adata.get("streams"):
            has_audio = True
            audio_codec = adata["streams"][0].get("codec_name")

    return VideoInfo(
        width=stream["width"],
        height=stream["height"],
        fps=fps,
        total_frames=total_frames,
        duration=float(fmt.get("duration", stream.get("duration", 0)) or 0),
        codec=stream.get("codec_name", "unknown"),
        has_audio=has_audio,
        audio_codec=audio_codec,
        pixel_format=stream.get("pix_fmt", "yuv420p")
    )


def calculate_target_resolution(
    src_w: int, src_h: int, resolution_spec: str
) -> Tuple[int, int, float]:
    """
    Calculate target resolution maintaining aspect ratio.
    Returns (out_w, out_h, scale_factor).
    """
    if resolution_spec in RESOLUTION_MAP:
        target_w, target_h = RESOLUTION_MAP[resolution_spec]
    elif "x original" in resolution_spec.lower():
        mult = float(resolution_spec.lower().split("x")[0].strip())
        target_w, target_h = int(src_w * mult), int(src_h * mult)
    else:
        try:
            parts = resolution_spec.lower().split("x")
            target_w, target_h = int(parts[0].strip()), int(parts[1].strip())
        except (ValueError, IndexError):
            raise ValueError(f"Invalid resolution: {resolution_spec}")

    # Even dimensions only
    target_w = target_w // 2 * 2
    target_h = target_h // 2 * 2

    scale_w = target_w / src_w
    scale_h = target_h / src_h
    scale = min(scale_w, scale_h)

    out_w = int(src_w * scale) // 2 * 2
    out_h = int(src_h * scale) // 2 * 2
    return out_w, out_h, scale


def auto_tile_size(video_w: int, video_h: int, device: str) -> int:
    """Auto-determine tile size based on resolution and device."""
    area = video_w * video_h
    if device == "cpu":
        thresholds = [
            (480 * 360, 0),
            (1280 * 720, 256),
            (1920 * 1080, 256),
            (3840 * 2160, 128),
            (float('inf'), 128)
        ]
    else:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            vram_gb = 0
        if vram_gb >= 16:
            thresholds = [
                (480 * 360, 0), (1280 * 720, 512),
                (1920 * 1080, 512), (3840 * 2160, 384), (float('inf'), 256)
            ]
        elif vram_gb >= 8:
            thresholds = [
                (480 * 360, 0), (1280 * 720, 512),
                (1920 * 1080, 384), (3840 * 2160, 256), (float('inf'), 256)
            ]
        else:
            thresholds = [
                (480 * 360, 0), (1280 * 720, 384),
                (1920 * 1080, 256), (3840 * 2160, 128), (float('inf'), 128)
            ]

    for max_area, tile in thresholds:
        if area <= max_area:
            return tile
    return 256


def get_upsampler(model_name: str, tile: int, device: str):
    """Create RealESRGANer upsampler."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    cfg = MODELS[model_name]
    weights_dir = REAL_ESRGAN_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if cfg["class"] == "RRDBNet":
        model = RRDBNet(**cfg["params"])
    else:
        model = SRVGGNetCompact(**cfg["params"])

    netscale = cfg["netscale"]
    file_url = cfg["url"]

    if isinstance(file_url, list):
        model_paths = [load_file_from_url(url=u, model_dir=str(weights_dir),
                                          progress=True, file_name=None) for u in file_url]
        model_path = model_paths
        dni_weight = [0.5, 0.5]
    else:
        model_path = load_file_from_url(url=file_url, model_dir=str(weights_dir),
                                        progress=True, file_name=None)
        dni_weight = None

    # Half precision: only on CUDA with compute capability >= 5.3
    use_half = False
    if device == "cuda" and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major > 5 or (major == 5 and minor >= 3):
            use_half = True

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
    return upsampler, netscale


# ---------------------------------------------------------------------------
# 6. Core upscaling function
# ---------------------------------------------------------------------------
def upscale_video(
    video_path: str,
    output_dir: str,
    resolution: str,
    model_name: str,
    device: str = "cpu",
    tile: int = 0,
    resume: bool = False,
    keep_temp: bool = False,
    crf: int = 18,
    preset: str = "slow",
    dry_run: bool = False,
) -> str:
    """
    Upscale a video to target resolution with audio preservation.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze
    print(f"[INFO] Analyzing: {video_path.name}")
    info = get_video_info(str(video_path))
    print(f"  {info.width}x{info.height} @ {info.fps:.2f}fps | "
          f"{info.total_frames} frames | {info.duration:.1f}s | "
          f"Audio: {'Yes (' + info.audio_codec + ')' if info.has_audio else 'No'}")

    # Target resolution
    out_w, out_h, scale = calculate_target_resolution(info.width, info.height, resolution)

    if scale < 1.0:
        print(f"[WARN] Downscale requested ({scale:.2f}x). Output will be smaller than input.")
    elif scale < 1.5:
        print(f"[WARN] Minimal upscaling ({scale:.2f}x). Quality gain may be limited.")

    print(f"[INFO] Target: {out_w}x{out_h} (scale: {scale:.3f}x)")

    video_base = video_path.stem
    final_output = output_dir / f"{video_base}_upscaled_{out_w}x{out_h}.mp4"

    if resume and final_output.exists():
        print(f"[SKIP] Already exists: {final_output}")
        return str(final_output)

    if dry_run:
        print(f"[DRY-RUN] Would output to: {final_output}")
        return str(final_output)

    # Auto tile
    if tile == 0:
        tile = auto_tile_size(info.width, info.height, device)
        if tile:
            print(f"[INFO] Auto tile size: {tile}")

    # Load model
    print(f"[INFO] Loading model: {model_name}...")
    upsampler, netscale = get_upsampler(model_name, tile, device)

    # Temp directory
    temp_dir = Path(tempfile.gettempdir()) / f"upscale_{video_base}_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)
    temp_raw = temp_dir / "frames.raw"

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV failed to open video")

        raw_fh = open(temp_raw, "wb")
        pbar = tqdm(total=info.total_frames, desc="Upscaling", unit="frame")
        failed = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_rgb, _ = upsampler.enhance(frame_rgb, outscale=scale)
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

                if out_bgr.shape[1] != out_w or out_bgr.shape[0] != out_h:
                    out_bgr = cv2.resize(out_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                raw_fh.write(out_bgr.tobytes())
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"\n[WARN] Frame error: {e}")
                fb = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                raw_fh.write(fb.tobytes())

            pbar.update(1)
            idx = pbar.n
            if idx % 30 == 0:
                elapsed = time.time() - t0
                speed = idx / elapsed if elapsed else 0
                eta = (info.total_frames - idx) / speed if speed else 0
                pbar.set_postfix({"fps": f"{speed:.1f}", "eta": f"{eta:.0f}s"})

        cap.release()
        raw_fh.close()
        pbar.close()

        if failed:
            print(f"[WARN] {failed}/{info.total_frames} frames used fallback resize")

        # Encode with ffmpeg
        print(f"[INFO] Encoding (CRF {crf}, preset {preset})...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
            "-r", str(info.fps), "-i", str(temp_raw),
        ]
        if info.has_audio:
            ffmpeg_cmd += ["-i", str(video_path), "-c:a", "copy", "-shortest"]
        else:
            ffmpeg_cmd += ["-an"]

        ffmpeg_cmd += [
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-colorspace", "bt709",
            "-color_trc", "bt709",
            "-color_primaries", "bt709",
            str(final_output)
        ]

        run_cmd(ffmpeg_cmd)

        size_mb = final_output.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] {final_output} ({size_mb:.1f} MB)")
        return str(final_output)

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"[INFO] Temp kept: {temp_dir}")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def main():
    model_help = "\n".join(f"  {k:<30} {v['description']}" for k, v in MODELS.items())
    res_help = "\n".join(f"  {k:<30} {v[0]}x{v[1]}" for k, v in RESOLUTION_MAP.items())

    parser = argparse.ArgumentParser(
        description="4K Video Upscaler - Enhanced Edition v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Models:
{model_help}

Resolutions:
{res_help}
  2x original                    Double original resolution
  4x original                    Quadruple original resolution
  WIDTHxHEIGHT                   Custom, e.g. 1920x1080
"""
    )
    parser.add_argument("--input", "-i", required=True, help="Input video or directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--resolution", "-r", default="4k (3840 x 2160)", help="Target resolution")
    parser.add_argument("--model", "-m", default="RealESRGAN_x4plus", choices=list(MODELS.keys()))
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0=auto)")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg CRF (18=visually lossless)")
    parser.add_argument("--preset", default="slow", choices=["ultrafast","superfast","veryfast",
                        "faster","fast","medium","slow","slower","veryslow"])
    parser.add_argument("--resume", action="store_true", help="Skip existing outputs")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--batch", action="store_true", help="Process all videos in directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline without processing")

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")

    inp = Path(args.input)
    targets = []
    if args.batch and inp.is_dir():
        targets = [f for f in inp.iterdir()
                   if f.suffix.lower() in VIDEO_EXTS]
        if not targets:
            print(f"[ERROR] No videos found in {inp}")
            sys.exit(1)
        print(f"[INFO] Batch mode: {len(targets)} video(s)")
    else:
        if not inp.exists():
            print(f"[ERROR] Not found: {inp}")
            sys.exit(1)
        targets = [inp]

    for i, vid in enumerate(targets, 1):
        if len(targets) > 1:
            print(f"\n{'='*60}\n[{i}/{len(targets)}] {vid.name}\n{'='*60}")
        try:
            upscale_video(str(vid), args.output, args.resolution, args.model,
                          device, args.tile, args.resume, args.keep_temp,
                          args.crf, args.preset, args.dry_run)
        except Exception as e:
            print(f"[ERROR] {vid.name}: {e}")
            if not args.batch:
                raise


if __name__ == "__main__":
    main()
