
# Gestura 🤟

**Real-Time Norwegian Sign Language Recognition System**

Gestura is an open-source computer vision system that recognizes Norwegian Sign Language hand gestures and converts them into typed text in real-time. Built with OpenCV and Python, it supports all 29 Norwegian letters (A-Å) plus SPACE and DELETE gestures.

![Gestura Demo](demo.gif)
*Real-time gesture recognition with skeleton overlay*

---

## ✨ Features

- **Norwegian Sign Language Support** - All 29 letters including Æ, Ø, Å
- **Real-Time Recognition** - 30+ FPS with sub-100ms latency
- **Dual Skeleton Tracking** - CV mode (classical algorithms) or MediaPipe mode (deep learning)
- **Offline Operation** - No internet connection required
- **Adaptive Calibration** - Automatic or manual HSV tuning for different skin tones and lighting
- **System-Level Typing** - Works with any application (text editors, browsers, etc.)
- **Visual Feedback** - Live skeleton overlay, confidence meters, and subtitle display

---

## 🎯 Quick Start

### Prerequisites

- Python 3.11+
- Webcam (720p recommended)
- Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BILYYY/gestura.git
   cd gestura
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional - Install MediaPipe for better accuracy:**
   ```bash
   pip install mediapipe
   ```

### Run Gestura

```bash
python run_gestura.py
```

On first launch:
1. Choose skeleton mode (1=CV, 2=MediaPipe)
2. Press `1` for simple calibration or `S` to skip
3. Show letter "A" and press SPACE when ready
4. Press `A` to activate typing mode
5. Start signing! 🤟

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **A** | Toggle typing mode (ACTIVE/INACTIVE) |
| **K** | Re-run calibration |
| **H** | Toggle help overlay |
| **G** | Toggle guide box |
| **ESC** | Quit |

---

## 📸 How It Works

```
Camera → Hand Detection → Skeleton Tracking → Feature Extraction → Recognition → Typing
         (HSV + Morphology)  (CV/MediaPipe)     (ORB + Geometric)    (Temporal Filter)
```

### Recognition Pipeline

1. **Hand Detection** - HSV color space filtering + morphological operations
2. **Skeleton Tracking** - Fingertip detection (convex hull or MediaPipe)
3. **Feature Extraction** - Geometric features + ORB texture + skeleton data
4. **Hybrid Fusion** - Combines 40% geometric + 40% ORB + 20% skeleton
5. **Temporal Stabilization** - 3-tier filtering eliminates flickering
6. **Keyboard Output** - System-level typing via pynput

---

## 📊 Performance

| Metric | CV Mode | MediaPipe Mode |
|--------|---------|----------------|
| **Overall Accuracy** | 85-90% | 94-96% |
| **Close Fingers (M,N,W)** | 60-70% | 98-99% |
| **Frame Rate** | 34-36 FPS | 28-32 FPS |
| **Latency** | ~50ms | ~60ms |

*Tested on Intel i5 8th Gen CPU*

---

## 🛠️ Project Structure

```
gestura/
├── gestura/                   # Core modules
│   ├── hand_detector.py       # Hand detection (HSV + morphology)
│   ├── recognizer_orb.py      # Gesture recognition (hybrid fusion)
│   ├── keyboard_manager.py    # Keyboard output
│   ├── subtitle_manager.py    # Visual feedback
│   └── calibration.py         # Calibration wizard
├── tools/
│   ├── _skeleton.py           # Skeleton tracking (CV + MediaPipe)
│   ├── _shared_utils.py       # Helper functions
│   └── capture.py             # Reference capture tool
├── resources/
│   ├── references/            # Universal reference images
│   └── references_personal/   # Personal reference images
├── requirements.txt
└── run_gestura.py            # Main application
```

---

## 🎨 Customization

### Capture Personal References

For better accuracy with your specific hand:

```bash
python tools/capture.py
```

Follow the 4-phase process:
1. Calibrate
2. Test quality
3. Choose "Personal" mode (press `2`)
4. Capture all letters A-Å + SPACE + DELETE

Personal references save to `resources/references_personal/`

### Manual Calibration

If automatic calibration fails:
1. Press `K` during main interface
2. Press `M` for manual HSV tuning
3. Adjust 6 sliders while watching live preview
4. Press SPACE to save

---

## 🐛 Troubleshooting

### Hand Not Detected
- ✅ Re-run calibration (press `K`)
- ✅ Try manual HSV tuning (press `M`)
- ✅ Improve lighting (avoid backlighting)
- ✅ Use plain, neutral background

### Wrong Letters Recognized
- ✅ Capture personal references (`python tools/capture.py`)
- ✅ Enable MediaPipe mode for letters M, N, W
- ✅ Hold gesture steady (wait for confidence >70%)
- ✅ Check gesture matches reference alphabet

### Slow Performance
- ✅ Use CV mode instead of MediaPipe
- ✅ Close other camera applications
- ✅ Lower camera resolution

---

## 📚 Documentation

Full documentation including system architecture, implementation details, and evaluation results is available in the `docs/` folder.

- [Installation Guide](docs/installation.md)
- [User Manual](docs/user_manual.md)
- [API Documentation](docs/api.md)
- [Technical Report](docs/report.pdf)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Multi-hand support
- Dynamic gesture recognition (motion-based signs)
- Additional sign languages
- Mobile platform ports
- Performance optimizations
- Bug fixes

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👥 Authors

- **[Elias Bouchabti ]** - [GitHub](https://github.com/BILYYY)
- **[Marthe]**
- **[Larsh]**
- **[Rafeal]**

---

## 🙏 Acknowledgments

- Norwegian Sign Language alphabet reference from [source]
- OpenCV community for excellent documentation
- MediaPipe team for hand landmark detection
- Course instructor for guidance and feedback

---


## ⚠️ Known Limitations

- Single-hand recognition only (no two-handed signs)
- Static gestures only (no motion-based signs)
- Requires calibration per environment
- Hand must be roughly upright (±30° rotation)
- Success rate ~75% (depends on lighting and camera quality)

---

## 🗺️ Roadmap

- [ ] Mobile app (Android/iOS)
- [ ] Multi-hand support
- [ ] Dynamic gesture recognition
- [ ] Word prediction/autocomplete
- [ ] Additional sign languages (ASL, BSL, etc.)
- [ ] Deep learning end-to-end model
- [ ] Web-based version (WebAssembly)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=BILYYY/gestura&type=Date)](https://star-history.com/#BILYYY/gestura&Date)

---

<div align="center">

**Made with ❤️ for the Norwegian deaf and hard-of-hearing community**

[Report Bug](https://github.com/BILYYY/gestura/issues) · [Request Feature](https://github.com/BILYYY/gestura/issues) · [Documentation](docs/)
