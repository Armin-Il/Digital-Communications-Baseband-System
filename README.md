# 📡 Digital Communications Baseband System

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Python implementation of a digital communications baseband system —
speech processing, uniform quantization, μ-Law companding, PCM
encoding, Polar NRZ/RZ/Manchester line coding, AWGN channel
simulation, BER analysis, and signal reconstruction.

## 📖 Project Overview

This project implements the fundamental stages of a digital
communications baseband system in Python. A recorded speech signal is
processed through source coding and line coding, then transmitted over
a simulated AWGN channel. Simulated BER is compared against the
theoretical BER, and reconstructed signal quality is evaluated both
numerically (SQNR, MSE) and by listening to the recovered audio.

## ✨ Features

- 🎤 Speech acquisition from a WAV file (16 kHz, mono, 16-bit PCM)
- 📊 Uniform quantization (2, 4, and 8 bits)
- 📈 μ-Law companding (compression + expansion, μ = 255)
- 🔢 Natural-binary PCM bit-stream encoding
- 📡 Polar NRZ, Polar RZ, and Manchester line coding
- 📉 BER simulation vs. Eb/N0, compared against theoretical Q-function BER
- 🌊 Power Spectral Density (PSD) analysis and bandwidth estimation
- 🔊 End-to-end signal reconstruction
- 📂 Automatic figure generation (histograms, PSD, BER curves)

## 🔄 System Pipeline

```
Speech.wav
    │
    ▼
Preprocessing (mono, 16 kHz, normalized to [-1, +1])
    │
    ▼
Uniform Quantization  ──┐
    │                   │
    ▼                   │
μ-Law Companding  ◄─────┘   (best quantizer selected by SQNR)
    │
    ▼
PCM Bit-Stream Encoding
    │
    ▼
Line Coding (NRZ / RZ / Manchester)
    │
    ▼
AWGN Channel
    │
    ▼
Matched Filter + Detection
    │
    ▼
BER Evaluation (simulated vs. theoretical)
    │
    ▼
Reconstructed Audio
```

## 📊 Experimental Results

| Item                   | Result         |
|------------------------|----------------|
| Sampling rate          | 16 kHz         |
| Speech duration        | 11.63 s        |
| Total bits             | 1,488,216      |
| Best quantizer         | μ-Law (8-bit)  |
| SQNR (8-bit uniform)   | 32.15 dB       |
| SQNR (8-bit μ-Law)     | 37.81 dB       |

The simulated BER closely tracks the theoretical AWGN BER curve
(within ±1 dB) across the tested Eb/N0 range.

## 🖼️ Figures

| BER vs. Eb/N0 | PSD | Quantization Error Histogram |
|:---:|:---:|:---:|
| ![BER](Figures/ber_curve.png) | ![PSD](Figures/psd.png) | ![Histogram](Figures/histogram.png) |

## 📁 Repository Structure

```
Digital-Communications-Baseband-System/
├── Figures/            # generated plots (BER, PSD, histograms, ...)
├── main.py             # entry point — runs the full pipeline
├── log_report.txt       # numeric results from the last run
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 🚀 Installation & Usage

```bash
git clone https://github.com/Armin-Il/Digital-Communications-Baseband-System.git
cd Digital-Communications-Baseband-System
pip install -r requirements.txt
python main.py path/to/your_speech.wav
```

If no WAV file is provided, the script falls back to a synthetic test
signal so the pipeline can still be verified end to end.

## 📦 Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

## 🔮 Future Improvements

- Digital modulation (BPSK, QPSK, QAM) and pulse shaping (RC/RRC)
- Realistic channel effects: multipath, frequency/phase offset
- Carrier recovery and adaptive modulation
- Channel coding (e.g., convolutional / LDPC)
- Rayleigh fading channel
- OFDM
- Symbol timing synchronization

## 👨‍💻 Author

**Armin Ilat**
Electrical Engineering Student, Shahid Beheshti University

## 📄 License

Released under the [MIT License](LICENSE).
