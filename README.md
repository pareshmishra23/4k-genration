# 🎬 4K Video Upscaler (Real-ESRGAN) v2.0

Upscale your videos up to **4K / 8K** on free Google Colab or locally using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

 
---

## ✨ What's New in v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Audio** | ❌ Stripped | ✅ Preserved via FFmpeg stream copy |
| **Compression** | Double (mp4v → libx264) | Single pass raw → libx264 |
| **Auto-tiling** | Manual only | Smart auto by resolution + VRAM |
| **Batch mode** | ❌ | ✅ Process entire folders |
| **Resume** | ❌ | ✅ Skip already-done files |
| **Color space** | Untagged | BT.709 tagged |
| **Error recovery** | Crash on bad frame | Lanczos fallback per frame |
| **Progress** | Basic bar | Live FPS + ETA |
| **basicsr patch** | Manual | Auto on import |
| **Dry run** | ❌ | ✅ Validate without processing |
| **Web playback** | Slow start | Faststart moov atom |
| **Quality control** | Fixed | Configurable CRF + preset |

---

## 🚀 Quick Start (Google Colab)

1. Open the notebook: **[4k_Video_Upscaler_Colab_(Real_ESRGAN).ipynb](https://github.com/pareshmishra23/4k-genration/blob/main/4k_Video_Upscaler_Colab_(Real_ESRGAN).ipynb)**
2. Change runtime to GPU: `Runtime` → `Change runtime type` → `T4 GPU`
3. Run all cells → Upload video → Configure → Upscale → Download

---

## 🖥️ Local Installation

```bash
# 1. Clone
git clone https://github.com/pareshmishra23/4k-genration.git
cd 4k-genration

# 2. Install Python deps
pip install -r requirements.txt

# 3. Install Real-ESRGAN
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN && pip install . && cd ..

# 4. Upscale!
python upscale_video.py -i video.mp4 -o ./output -r "4k (3840 x 2160)"
```

---

## 📋 CLI Usage

```bash
# Single video
python upscale_video.py -i video.mp4 -o ./output -r "4k (3840 x 2160)" -m RealESRGAN_x4plus

# Batch folder
python upscale_video.py -i ./videos/ --batch -o ./output -r "4k (3840 x 2160)" --resume

# Fast draft quality
python upscale_video.py -i video.mp4 --preset fast --crf 23

# Validate setup without processing
python upscale_video.py -i video.mp4 --dry-run
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input video or directory | **required** |
| `-o, --output` | Output directory | `./output` |
| `-r, --resolution` | Target resolution | `4k (3840 x 2160)` |
| `-m, --model` | Model name | `RealESRGAN_x4plus` |
| `--device` | `cuda` or `cpu` | auto-detect |
| `--tile` | Tile size (0 = auto) | `0` |
| `--crf` | FFmpeg CRF (lower = better) | `18` |
| `--preset` | Encode speed/quality tradeoff | `slow` |
| `--batch` | Process all videos in folder | — |
| `--resume` | Skip existing outputs | — |
| `--dry-run` | Validate without processing | — |
| `--keep-temp` | Keep temporary files | — |

---

## 🧠 Models

| Model | Best For | Speed |
|-------|----------|-------|
| `RealESRGAN_x4plus` | General photos/videos | Medium |
| `RealESRNet_x4plus` | General, fewer artifacts | Medium |
| `RealESRGAN_x4plus_anime_6B` | Anime/cartoons | Medium |
| `realesr-animevideov3` | Anime videos | **Fast** |
| `RealESRGAN_x2plus` | 2× upscaling only | Medium |
| `realesr-general-x4v3` | General + noise reduction | Medium |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Out of Memory | Lower `--tile` (128/256) or use CPU |
| No audio in output | Check source has audio; v2.0 auto-preserves |
| Very slow | Use `--preset fast` or `--preset medium` |
| Wrong colors | v2.0 tags BT.709; try different model |
| `basicsr` import error | Restart runtime; v2.0 auto-patches on load |

---

## 📄 License

MIT

## 🙏 Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- Original Colab structure by [yuvraj108c](https://github.com/yuvraj108c)
- Enhanced v2.0 by [pareshmishra23](https://github.com/pareshmishra23)
