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

 
**Phase 1 — Baseband**
- 🎤 Speech acquisition from a WAV file (16 kHz, mono, 16-bit PCM)
- 📊 Uniform quantization (2, 4, and 8 bits)
- 📈 μ-Law companding (compression + expansion, μ = 255)
- 🔢 Natural-binary PCM bit-stream encoding
- 📡 Polar NRZ, Polar RZ, and Manchester line coding
- 📉 BER simulation vs. Eb/N0, compared against theoretical Q-function BER
- 🌊 Power Spectral Density (PSD) analysis and bandwidth estimation
- 🔊 End-to-end signal reconstruction
- 📂 Automatic figure generation (histograms, PSD, BER curves)

**Phase 2 — Passband**
- 📶 Digital modulation: BPSK, QPSK, 16-QAM, 64-QAM (Gray-coded)
- 🌀 Pulse shaping: rectangular, raised cosine (RC), and root-raised cosine (RRC)
- 📡 AWGN channel with frequency offset and phase offset impairments
- 🔒 Pilot-aided carrier recovery via a Costas loop
- 📉 BER simulation vs. SNR (with/without carrier recovery) vs. theoretical BER
- ⚙️ Adaptive modulation: SNR-based scheme selection with throughput comparison
- 🎙️ End-to-end voice transmission through the modulated channel and reconstruction
- 🖼️ Constellation diagrams, eye diagrams, BER curves, and throughput plots


## 🔄 System Pipeline

**Phase 1**
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
μ-Law Companding  ◄─────┘   (best configuration selected by SQNR)
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
**Phase 2**
```
PCM Bit-Stream (from speech, μ-law encoded)
    │
    ▼
Digital Modulation (BPSK / QPSK / 16-QAM / 64-QAM)
    │
    ▼
Pulse Shaping (Rect / RC / RRC)
    │
    ▼
Channel (Freq Offset + Phase Offset + AWGN)
    │
    ▼
Matched Filter + Symbol Sampling
    │
    ▼
Pilot-Aided Carrier Recovery (Costas Loop)
    │
    ▼
Demodulation
    │
    ▼
BER Evaluation (simulated vs. theoretical) + Adaptive Modulation
    │
    ▼
Reconstructed Audio
```

## 📊 Experimental Results

**Phase 1**

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
 
**Phase 2**
 
Simulated BER (with pilot-aided carrier recovery) closely tracks the
theoretical BER curve for the selected modulation scheme across the tested
SNR range; without carrier recovery, residual frequency/phase offset causes
a visible error floor. Adaptive modulation switches scheme with SNR
(BPSK → QPSK → 16-QAM → 64-QAM) to maximize throughput. Full plots are in
`results_phase2/`.

## 📁 Repository Structure

```
Digital-Communications-Baseband-System/
├── Figures/            # Phase 1 plots (BER, PSD, histograms, ...)
├── results_phase2/     # Phase 2 plots (constellation, eye diagram, BER, throughput)
├── main.py             # Phase 1 entry point — runs the baseband pipeline
├── phase2.py            # Phase 2 entry point — runs the passband pipeline
├── log_report.txt       # numeric results from the last Phase 1 run
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

- Channel coding (e.g., convolutional / LDPC)
- Rayleigh fading channel
- OFDM
- Symbol timing synchronization

## 👨‍💻 Author

**Armin Ilat**
Electrical Engineering Student, Shahid Beheshti University

## 📄 License

Released under the [MIT License](LICENSE).
