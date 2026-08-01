# 4K Video Upscaler (Real-ESRGAN)

Upscale your videos up to 4K using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

This is a corrected, standalone Python version adapted from the original [Google Colab notebook](https://github.com/yuvraj108c/4k-video-upscaler-colab) by [yuvraj108c](https://github.com/yuvraj108c).

## Features

- **CPU and GPU support** — Automatically detects available hardware and falls back to CPU if no GPU is present
- **Multiple models** — Choose from various Real-ESRGAN models for different content types
- **Flexible resolution** — Upscale to FHD (1920x1080), 2K (2560x1440), 4K (3840x2160), or multiples of original
- **Tile processing** — Use `--tile` to reduce memory usage for large videos

## Models

| Model | Best For |
|-------|----------|
| `RealESRGAN_x4plus` | General photos/videos |
| `RealESRGAN_x4plus_anime_6B` | Anime/cartoon content |
| `realesr-animevideov3` | Anime videos (recommended, fastest) |
| `RealESRNet_x4plus` | General (no artifacts) |
| `RealESRGAN_x2plus` | General (2x upscale) |
| `realesr-general-x4v3` | General (4x, with denoise) |

## Installation

```bash
# Clone the repository
git clone https://github.com/pareshmishra23/4k-genration.git
cd 4k-genration

# Clone Real-ESRGAN (required dependency)
git clone https://github.com/xinntao/Real-ESRGAN.git

# Install dependencies
pip install -r requirements.txt
pip install -e Real-ESRGAN
```

## Usage

### Basic Usage

```bash
python upscale_video.py --input video.mp4 --output ./output --resolution "4k (3840 x 2160)"
```

### All Options

```bash
python upscale_video.py \
  --input video.mp4 \
  --output ./output \
  --resolution "4k (3840 x 2160)" \
  --model RealESRGAN_x4plus \
  --device auto \
  --tile 0
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | *(required)* | Input video file path |
| `--output`, `-o` | `./output` | Output directory |
| `--resolution` | `4k (3840 x 2160)` | Target resolution |
| `--model` | `RealESRGAN_x4plus` | Model to use |
| `--device` | `auto` | Device: `cuda`, `cpu`, or `auto` |
| `--tile` | `0` | Tile size (0=auto, smaller for less VRAM) |
| `--gpu` | Flag | Force GPU usage |

### Resolution Options

- `FHD (1920 x 1080)`
- `2k (2560 x 1440)`
- `4k (3840 x 2160)`
- `2 x original`
- `3 x original`
- `4 x original`

### Examples

```bash
# Upscale to FHD using anime model (fastest)
python upscale_video.py -i video.mp4 -o ./output --resolution "FHD (1920 x 1080)" --model realesr-animevideov3

# Upscale to 4K using general model on GPU
python upscale_video.py -i video.mp4 -o ./output --resolution "4k (3840 x 2160)" --model RealESRGAN_x4plus --gpu

# Upscale to 4x original on CPU with tile processing (low memory)
python upscale_video.py -i video.mp4 -o ./output --resolution "4 x original" --tile 64 --device cpu
```

## Performance Tips

- **GPU with CUDA** is recommended for best performance (Colab T4/V100, local GPU)
- **Tile size** (`--tile`): Use smaller values (e.g., 64, 128) for low VRAM GPUs or CPU
- **Model choice**: `realesr-animevideov3` is the fastest; `RealESRGAN_x4plus` produces the highest quality
- **Resolution**: Upscaling to 4K on CPU may take several minutes per frame

## Original Project

This project is a corrected standalone adaptation of:
- [4k-Video-Upscaler-Colab](https://github.com/yuvraj108c/4k-video-upscaler-colab) by [yuvraj108c](https://github.com/yuvraj108c)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by [xinntao](https://github.com/xinntao)

## License

MIT
