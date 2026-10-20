# Mixxx Stems

[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/dromangdev/mixxx-stems)
[![Engine](https://img.shields.io/badge/AI%20Engine-HTDemucs%20v4%20%2F%20FT-orange.svg)](https://github.com/dromangdev/mixxx-stems)
[![ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime%20C%2B%2B-blueviolet.svg)](https://github.com/microsoft/onnxruntime)

**Mixxx Stems** is a high-performance fork of [Mixxx 2.5](https://mixxx.org) featuring **Real-Time Deep Neural Network AI Stem Separation** (*Vocals*, *Drums*, *Instruments*). Powered by Meta AI's **HTDemucs v4** & **HTDemucs-FT** and Microsoft's **ONNX Runtime C++**.

---

## ✨ Features

- **🧠 Real Neural Network Separation:** True acoustic source separation (not simple EQ frequency cuts). Separates complex mixes into pristine Vocals, Drums, and Instruments.
- **⚡ 100% Offline & Private:** Runs entirely on local hardware (CPU / GPU / Apple Neural Engine). No internet connection or cloud API is needed to DJ.
- **🔁 Overlap-Add Crossfade (OLA):** 25% overlap processing with linear trapezoidal windowing eliminates clicks and phase jumps between chunk boundaries.
- **💾 VirtualDJ-Style Disk Cache (`.stems`):** Once a track is processed, stems are cached locally for instant loading (<50 ms) on all future sessions.
- **🎛️ Integrated LateNight Skin Controls:** Dedicated per-deck controls for **Mute**, **Solo**, **Gain Faders**, and **Reset** for each individual stem.
- **🛡️ Zero-Latency DSP Fallback:** 4th-order Linkwitz-Riley (24 dB/oct) crossover filter automatically engages if neural network models are absent.
- **📦 Zero-Setup Dependency Fetch:** CMake automatically downloads and links prebuilt ONNX Runtime libraries for your OS and architecture during build if not already installed.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/dromangdev/mixxx-stems.git
cd mixxx-stems
```

### 2. Download the AI Models

Place the model weights into the `models/` directory:

```bash
mkdir -p models
# Dual specialist models (Recommended for highest fidelity):
curl -L -o models/htdemucs_ft_vocals.onnx "https://huggingface.co/StemSplitio/htdemucs-ft-onnx/resolve/main/htdemucs_ft_vocals.onnx"
curl -L -o models/htdemucs_ft_drums.onnx  "https://huggingface.co/StemSplitio/htdemucs-ft-onnx/resolve/main/htdemucs_ft_drums.onnx"
```

*(See [models/README.md](models/README.md) for alternative models like standard HTDemucs v4, SCNet, or BS-RoFormer).*

### 3. Build & Run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Run Mixxx
./build/mixxx
```

> **Note:** ONNX Runtime is automatically detected or downloaded by CMake during the configure step. No manual package installation required.

---

## 🎚️ How to Use in Mixxx

1. Open **Mixxx** and load any song into **Deck 1** or **Deck 2**.
2. In the **LateNight** skin, locate the **STEMS** section under each deck's waveform.
3. Use the **VOCALS**, **DRUMS**, and **INST** buttons:
   - **Mute (`M`):** Cut out vocals or drums on the fly for live mashups.
   - **Solo (`S`):** Isolate an acapella or drum break instantly.
   - **Faders:** Adjust relative volume balance between stems.
   - **Reset (`R`):** Instantly restore original track balance.

---

## 📂 Model Storage Paths

The engine automatically scans the following locations for `.onnx` models:
- `./models/` (Local repository root)
- Application directory (`<exe_path>/models/` or macOS app bundle resources)
- **macOS:** `~/Library/Application Support/Mixxx/models/`
- **Linux:** `~/.local/share/Mixxx/models/` or `/usr/share/mixxx/models/`
- **Windows:** `%LOCALAPPDATA%\Mixxx\models\`

---

## 📄 License

Mixxx Stems is licensed under the [GNU General Public License v2 (GPL-2.0)](LICENSE).
