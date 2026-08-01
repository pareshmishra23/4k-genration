# 4K Video Upscaler (Real-ESRGAN)

Upscale your videos up to 4K on free Google Colab or locally using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pareshmishra23/4k-genration/blob/main/4k_Video_Upscaler_Colab_(Real_ESRGAN).ipynb)

## Features

- **Multi-Environment Support** — Works in Google Colab and standard Linux/Windows Python environments.
- **Flexible Input** — Support for Google Drive, Direct Upload (Colab), or Local Paths.
- **Multiple Models** — Choose between general, anime, and fast video models.
- **High Resolution** — Upscale to FHD (1080p), 2K, 4K, or custom multipliers (2x, 3x, 4x).
- **Auto-Patching** — Automatically fixes common library compatibility issues (e.g., `basicsr` vs `torchvision`).

## Models

| Model | Description |
|-------|-------------|
| `RealESRGAN_x4plus` | Best for general photos and high-quality videos. |
| `RealESRGAN_x4plus_anime_6B` | Optimized for anime and cartoons. |
| `realesr-animevideov3` | Fastest model for anime videos. |
| `RealESRNet_x4plus` | General model with fewer artifacts. |
| `RealESRGAN_x2plus` | Optimized for 2x upscaling. |
| `realesr-general-x4v3` | General model with noise reduction. |

## Installation (Local)

```bash
# Clone the repository
git clone https://github.com/pareshmishra23/4k-genration.git
cd 4k-genration

# Install dependencies
pip install -r requirements.txt

# Clone Real-ESRGAN dependency
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN && pip install . && cd ..
```

## Usage (Local)

```bash
python upscale_video.py --input video.mp4 --output ./output --resolution "4k (3840 x 2160)" --model RealESRGAN_x4plus
```

### Options

- `--input`, `-i`: Path to the input video.
- `--output`, `-o`: Directory to save the result.
- `--resolution`, `-r`: Target resolution (e.g., "4k (3840 x 2160)", "2 x original").
- `--model`, `-m`: Model name to use.
- `--device`: `cuda` or `cpu` (default: auto-detect).
- `--tile`: Tile size for low-memory processing (default: 0 for auto).

## Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao.
- Original Colab notebook structure by [yuvraj108c](https://github.com/yuvraj108c).

## License

MIT
