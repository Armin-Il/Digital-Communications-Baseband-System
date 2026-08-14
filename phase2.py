import os
import sys

# configure terminal encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import numpy as np
from scipy import signal as sig
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

RNG = np.random.default_rng(7)


# core math and modulation functions

def q_function(x):
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2.0))

def _int_to_gray(n):
    return n ^ (n >> 1)

def _gray_to_int(g):
    n = g
    shift = 1
    while (g >> shift) > 0:
        n ^= (g >> shift)
        shift += 1
    return n

def _pam_levels(k, gray=True):
    M = 2 ** k
    idx = np.arange(M)
    mapping = np.array([_gray_to_int(i) for i in idx]) if gray else idx
    levels = 2 * mapping - (M - 1)
    return levels / (M - 1)

def bits_per_symbol(scheme):
    return {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6, "64qam_gray": 6}[scheme]

def _bits_to_int_groups(bits, k):
    n = len(bits) // k
    bits = np.asarray(bits[: n * k], dtype=np.uint32).reshape(n, k)
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.uint32)
    return bits.dot(weights)

def _int_groups_to_bits(vals, k):
    vals = np.asarray(vals, dtype=np.uint32)
    shifts = np.arange(k - 1, -1, -1)
    bits = ((vals[:, None] >> shifts) & 1).astype(np.uint8)
    return bits.flatten()

def modulate(bits, scheme):
    scheme = scheme.lower()

    if scheme == "bpsk":
        b = np.asarray(bits[: (len(bits) // 1) * 1], dtype=np.uint8)
        symbols = np.where(b == 1, 1.0, -1.0).astype(np.complex128)
        const = {0: -1.0 + 0j, 1: 1.0 + 0j}
        return symbols, const

    if scheme == "qpsk":
        k = 2
        vals = _bits_to_int_groups(bits, k)
        gray_to_phase = {0: np.pi / 4, 1: 3 * np.pi / 4, 3: 5 * np.pi / 4, 2: 7 * np.pi / 4}
        phases = np.array([gray_to_phase[v] for v in vals])
        symbols = np.exp(1j * phases)
        const = {v: np.exp(1j * p) for v, p in gray_to_phase.items()}
        return symbols, const

    if scheme in ("16qam", "64qam", "64qam_gray"):
        use_gray = scheme.endswith("_gray")
        M = 16 if scheme == "16qam" else 64
        k_total = int(np.log2(M))
        k_half = k_total // 2
        vals = _bits_to_int_groups(bits, k_total)
        i_idx = (vals >> k_half) & ((1 << k_half) - 1)
        q_idx = vals & ((1 << k_half) - 1)
        pam = _pam_levels(k_half, gray=use_gray)
        I = pam[i_idx]
        Q = pam[q_idx]
        symbols = (I + 1j * Q)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

        side = 2 ** k_half
        const = {}
        for iv in range(side):
            for qv in range(side):
                v = (iv << k_half) | qv
                pt = (pam[iv] + 1j * pam[qv])
                const[v] = pt
        norm = np.sqrt(np.mean(np.abs(np.array(list(const.values()))) ** 2))
        const = {v: p / norm for v, p in const.items()}
        return symbols, const

    raise ValueError(f"unrecognized modulation scheme: {scheme}")

def demodulate(symbols, scheme):
    scheme = scheme.lower()
    _, const = modulate(np.zeros(bits_per_symbol(scheme) * 4, dtype=np.uint8), scheme)
    keys = np.array(list(const.keys()))
    pts = np.array(list(const.values()))
    d = np.abs(symbols[:, None] - pts[None, :])
    nearest = np.argmin(d, axis=1)
    vals = keys[nearest]
    k = bits_per_symbol(scheme)
    return _int_groups_to_bits(vals, k)

def get_constellation(scheme):
    _, const = modulate(np.zeros(bits_per_symbol(scheme) * 4, dtype=np.uint8), scheme)
    return const


def make_pilot_symbols(scheme, n_pilot, modulate_fn, seed=1234):
    bits_map = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6, "64qam_gray": 6}
    rng = np.random.default_rng(seed)
    k = bits_map[scheme.lower()]
    pilot_bits = rng.integers(0, 2, n_pilot * k).astype(np.uint8)
    pilot_symbols, _ = modulate_fn(pilot_bits, scheme)
    return pilot_bits, pilot_symbols

def carrier_recovery_costas(rx_symbols, known_pilot_symbols, scheme, get_constellation_fn,
                             loop_gain=0.02, pilot_gain_boost=2.0):
    n_pilot = len(known_pilot_symbols)
    const = get_constellation_fn(scheme)
    ref_pts = np.array(list(const.values())) if isinstance(const, dict) else np.asarray(const)

    # 1. Precise frequency offset estimation from the pilot using linear regression
    if n_pilot >= 8:
        phase_diff = np.angle(rx_symbols[:n_pilot] * np.conj(known_pilot_symbols))
        phase_unwrapped = np.unwrap(phase_diff)
        slope, intercept = np.polyfit(np.arange(n_pilot), phase_unwrapped, 1)
        est_delta_f = slope / (2 * np.pi)
        initial_phase = intercept
    else:
        est_delta_f = 0.0
        initial_phase = 0.0

    n_all = np.arange(len(rx_symbols))
    rx_fc = rx_symbols * np.exp(-1j * (2 * np.pi * est_delta_f * n_all + initial_phase))

    # 2. Costas loop for continuous phase tracking
    phase_est = 0.0
    corrected = np.zeros_like(rx_fc)
    for i in range(len(rx_fc)):
        s_corr = rx_fc[i] * np.exp(-1j * phase_est)
        corrected[i] = s_corr
        if i < n_pilot:
            ref = known_pilot_symbols[i]
            g = loop_gain * pilot_gain_boost
        else:
            idx = np.argmin(np.abs(s_corr - ref_pts))
            ref = ref_pts[idx]
            g = loop_gain
        phase_error = np.angle(s_corr * np.conj(ref))
        phase_est += g * phase_error

    return corrected, est_delta_f


def rect_pulse(sps, span=1):
    return np.ones(sps * span, dtype=np.float64)

def raised_cosine_pulse(r, sps, span=12):
    T = sps
    t = np.arange(-span * sps / 2, span * sps / 2 + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc_part = np.sinc(t / T)
        cos_part = np.cos(np.pi * r * t / T)
        denom = 1.0 - (2.0 * r * t / T) ** 2
        p = sinc_part * cos_part / denom
    if r > 0:
        t_sing = T / (2.0 * r)
        idx = np.isclose(np.abs(t), t_sing, atol=1e-6)
        p[idx] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * r))
    p[np.isnan(p)] = 1.0
    return p / np.sum(p) * sps

def root_raised_cosine_pulse(r, sps, span=12):
    T = float(sps)
    t = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    p = np.zeros_like(t)

    if r == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.sinc(t / T)
        return p / np.sqrt(np.sum(p ** 2)) * np.sqrt(sps)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            p[i] = (1.0 / T) * (1.0 + r * (4.0 / np.pi - 1.0))
        elif np.isclose(np.abs(ti), T / (4.0 * r), atol=1e-6):
            p[i] = (r / (T * np.sqrt(2.0))) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * r))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * r))
            )
        else:
            num = (np.sin(np.pi * ti / T * (1 - r)) +
                   4 * r * ti / T * np.cos(np.pi * ti / T * (1 + r)))
            den = (np.pi * ti / T) * (1 - (4 * r * ti / T) ** 2)
            p[i] = (1.0 / T) * num / den

    return p / np.sqrt(np.sum(p ** 2)) * np.sqrt(sps)

def get_pulse(kind, sps, r=0.35, span=12):
    kind = kind.lower()
    if kind == "rect":
        return rect_pulse(sps)
    if kind == "rc":
        return raised_cosine_pulse(r, sps, span)
    if kind == "rrc":
        return root_raised_cosine_pulse(r, sps, span)
    raise ValueError(kind)


def apply_freq_offset(x, delta_f_normalized, sps):
    n = np.arange(len(x))
    t_over_T = n / sps
    return x * np.exp(1j * 2 * np.pi * delta_f_normalized * t_over_T)

def apply_phase_offset(x, phi0):
    return x * np.exp(1j * phi0)

def awgn_complex(x, snr_db, rng=RNG):
    p_sig = np.mean(np.abs(x) ** 2)
    n0 = p_sig / (10 ** (snr_db / 10.0))
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise, n0

def apply_channel(x, sps, snr_db, delta_f_normalized=0.005, phi0=np.pi / 6, rng=RNG):
    y = apply_freq_offset(x, delta_f_normalized, sps)
    y = apply_phase_offset(y, phi0)
    y, n0 = awgn_complex(y, snr_db, rng)
    return y, n0

def matched_filter(rx_signal, pulse):
    mf_kernel = np.conj(pulse[::-1])
    return np.convolve(rx_signal, mf_kernel, mode="full")


def rx_pipeline(rx_signal, pulse, scheme, sps, n_total_symbols, pilot_symbols=None, do_carrier_recovery=True):
    rx_mf = matched_filter(rx_signal, pulse)
    pulse_delay = len(pulse) - 1
    sample_indices = pulse_delay + np.arange(n_total_symbols) * sps
    sample_indices = sample_indices[sample_indices < len(rx_mf)]

    rx_sampled = rx_mf[sample_indices]
    rx_sampled = rx_sampled / np.sqrt(np.mean(np.abs(rx_sampled) ** 2) + 1e-12)

    if do_carrier_recovery and pilot_symbols is not None:
        rx_recovered, _ = carrier_recovery_costas(
            rx_sampled, pilot_symbols, scheme, get_constellation
        )
    else:
        rx_recovered = rx_sampled

    rx_bits = demodulate(rx_recovered, scheme)
    return rx_bits, rx_recovered


def plot_constellation(scheme, out_path, noisy_symbols=None):
    const = get_constellation(scheme)
    k = bits_per_symbol(scheme)
    plt.figure(figsize=(5.5, 5.5))
    if noisy_symbols is not None:
        plt.scatter(noisy_symbols.real, noisy_symbols.imag, s=4, alpha=0.3,
                    color="gray", label="received noisy symbols")
    pts = np.array(list(const.values()))
    plt.scatter(pts.real, pts.imag, s=60, color="crimson", zorder=5, label="reference points")
    for v, p in const.items():
        label = format(v, f"0{k}b")
        plt.annotate(label, (p.real, p.imag), textcoords="offset points",
                     xytext=(6, 6), fontsize=7)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.title(f"Constellation Diagram - {scheme.upper()}")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    if noisy_symbols is not None:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

def build_eye_diagram_data(scheme, pulse_kind, sps, r=0.35, span=12, n_symbols=400, snr_db=None, rng=RNG):
    k = bits_per_symbol(scheme)
    bits = rng.integers(0, 2, n_symbols * k).astype(np.uint8)
    symbols, _ = modulate(bits, scheme)

    pulse = get_pulse(pulse_kind, sps, r, span)
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    tx = np.convolve(upsampled, pulse, mode="full")

    if snr_db is not None:
        p_sig = np.mean(np.abs(tx) ** 2)
        n0 = p_sig / (10 ** (snr_db / 10))
        noise = np.sqrt(n0 / 2) * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
        tx = tx + noise

    return tx.real

def plot_eye_diagram(wave, sps, out_path, title, n_windows=150, offset=0):
    plt.figure(figsize=(6, 4.5))
    two_T = 2 * sps
    n_avail = (len(wave) - offset) // two_T
    n_windows = min(n_windows, n_avail)
    t_axis = np.arange(two_T) / sps
    for i in range(n_windows):
        seg = wave[offset + i * two_T: offset + (i + 1) * two_T]
        if len(seg) < two_T:
            continue
        plt.plot(t_axis, seg, color="steelblue", alpha=0.25, linewidth=0.8)
    plt.title(title)
    plt.xlabel("time (t/T)")
    plt.ylabel("amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

def theoretical_ber(scheme, snr_db_list):
    snr_lin = 10 ** (np.array(snr_db_list, dtype=np.float64) / 10.0)
    scheme = scheme.lower()

    if scheme in ("bpsk", "qpsk"):
        return q_function(np.sqrt(2 * snr_lin))
    elif scheme == "16qam":
        return (3 / 4) * q_function(np.sqrt((4 / 5) * snr_lin))
    elif scheme in ("64qam", "64qam_gray"):
        return (7 / 12) * q_function(np.sqrt((2 / 7) * snr_lin))
    else:
        raise ValueError(f"modulation scheme not supported: {scheme}")

def run_ber_simulation(scheme, pulse_kind, snr_db_list, sps=8, n_data_symbols=4000, n_pilot=64, rng=RNG):
    k = bits_per_symbol(scheme)
    bits_tx = rng.integers(0, 2, n_data_symbols * k).astype(np.uint8)

    pilot_bits, pilot_symbols = make_pilot_symbols(scheme, n_pilot, modulate)
    data_symbols, _ = modulate(bits_tx, scheme)
    symbols_tx = np.concatenate([pilot_symbols, data_symbols])

    pulse = get_pulse(pulse_kind, sps=sps, r=0.35, span=12)

    tx_upsampled = np.zeros(len(symbols_tx) * sps, dtype=np.complex128)
    tx_upsampled[::sps] = symbols_tx
    tx_signal = np.convolve(tx_upsampled, pulse, mode="full")

    ber_no_rec = []
    ber_with_rec = []

    for snr_db in snr_db_list:
        rx_channel, _ = apply_channel(
            tx_signal, sps=sps, snr_db=snr_db,
            delta_f_normalized=0.005, phi0=np.pi / 6, rng=rng
        )

        # without carrier recovery
        bits_no_rec, _ = rx_pipeline(
            rx_channel, pulse, scheme, sps, len(symbols_tx), pilot_symbols=None, do_carrier_recovery=False
        )
        data_bits_no = bits_no_rec[n_pilot * k :]
        min_len1 = min(len(bits_tx), len(data_bits_no))
        err1 = np.mean(bits_tx[:min_len1] != data_bits_no[:min_len1])
        ber_no_rec.append(err1)

        # with carrier recovery
        bits_with_rec, _ = rx_pipeline(
            rx_channel, pulse, scheme, sps, len(symbols_tx), pilot_symbols=pilot_symbols, do_carrier_recovery=True
        )
        data_bits_with = bits_with_rec[n_pilot * k :]
        min_len2 = min(len(bits_tx), len(data_bits_with))
        err2 = np.mean(bits_tx[:min_len2] != data_bits_with[:min_len2])
        ber_with_rec.append(err2)

    return np.array(ber_no_rec), np.array(ber_with_rec)

def plot_ber_curves(scheme, snr_db_list, ber_no_rec, ber_with_rec, out_path):
    ber_theo = theoretical_ber(scheme, snr_db_list)

    plt.figure(figsize=(7, 5))
    plt.semilogy(snr_db_list, ber_no_rec, 'r--o', label="simulation (without carrier recovery)")
    plt.semilogy(snr_db_list, ber_with_rec, 'g-s', label="simulation (with carrier recovery)")
    plt.semilogy(snr_db_list, ber_theo, 'k:', linewidth=2, label="theory (ideal AWGN channel)")

    plt.title(f"Bit Error Rate (BER) vs SNR - {scheme.upper()}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.ylim(1e-5, 1.0)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

def select_adaptive_scheme(snr_db):
    if snr_db < 6.0:
        return "bpsk"
    elif 6.0 <= snr_db < 12.0:
        return "qpsk"
    elif 12.0 <= snr_db < 18.0:
        return "16qam"
    else:
        return "64qam_gray"

def plot_adaptive_throughput(snr_db_list, out_path):
    schemes = ["bpsk", "qpsk", "16qam", "64qam_gray"]
    throughput_fixed = {s: [] for s in schemes}
    throughput_adaptive = []

    for snr in snr_db_list:
        for s in schemes:
            k = bits_per_symbol(s)
            ber = theoretical_ber(s, [snr])[0]
            tp = k * (1.0 - ber)
            throughput_fixed[s].append(tp)

        best_s = select_adaptive_scheme(snr)
        k_adapt = bits_per_symbol(best_s)
        ber_adapt = theoretical_ber(best_s, [snr])[0]
        throughput_adaptive.append(k_adapt * (1.0 - ber_adapt))

    plt.figure(figsize=(7.5, 5))
    for s in schemes:
        plt.plot(snr_db_list, throughput_fixed[s], '--', label=f"fixed {s.upper()}")

    plt.plot(snr_db_list, throughput_adaptive, 'k-o', linewidth=2.5, label="adaptive modulation")
    plt.title("Data Throughput vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Throughput (bits/symbol)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def _mu_law_encode(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def _mu_law_decode(y, mu=255):
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu

def audio_to_bitstream(audio_samples, B=8):
    norm_audio = audio_samples / (np.max(np.abs(audio_samples)) + 1e-12)
    compressed = _mu_law_encode(norm_audio)
    levels = 2 ** B
    quantized = np.round(((compressed + 1) / 2) * (levels - 1)).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)
    return np.unpackbits(quantized)

def bitstream_to_audio(bits, B=8):
    quantized = np.packbits(bits)
    levels = 2 ** B
    norm_q = (quantized.astype(np.float64) / (levels - 1)) * 2.0 - 1.0
    return _mu_law_decode(norm_q)


def run_full_phase2_pipeline(out_dir="results_phase2"):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(" Starting project code execution - Armin Ilat 403249010")
    print("=" * 60)

    audio_filename = "Armin Ilat speech.wav"
    if os.path.exists(audio_filename):
        fs, audio_raw = wavfile.read(audio_filename)
        if audio_raw.dtype == np.int16:
            audio_orig = audio_raw.astype(np.float64) / 32768.0
        else:
            audio_orig = audio_raw.astype(np.float64)
        print(f"[+] Successfully read real audio file '{audio_filename}' at {fs}Hz")
    else:
        fs = 8000
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration))
        audio_orig = 0.6 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
        print(f"[!] File '{audio_filename}' not found, generated a synthetic signal instead")

    tx_bits = audio_to_bitstream(audio_orig, B=8)
    print(f"[+] Total bits extracted from the voice signal: {len(tx_bits)} bits")

    test_snr = 15.0
    scheme = select_adaptive_scheme(test_snr)
    sps = 8
    pulse_kind = "rrc"
    N_PILOT = 64

    print(f"[+] Based on SNR={test_snr}dB, selected modulation: {scheme.upper()}")

    pilot_bits, pilot_symbols = make_pilot_symbols(scheme, N_PILOT, modulate)
    data_symbols, _ = modulate(tx_bits, scheme)
    symbols_tx = np.concatenate([pilot_symbols, data_symbols])

    pulse = get_pulse(pulse_kind, sps=sps, r=0.35, span=12)

    tx_upsampled = np.zeros(len(symbols_tx) * sps, dtype=np.complex128)
    tx_upsampled[::sps] = symbols_tx
    tx_signal = np.convolve(tx_upsampled, pulse, mode="full")

    print("[+] Transmitting through channel (Freq Offset + Phase Offset + AWGN)...")
    rx_channel, _ = apply_channel(
        tx_signal, sps=sps, snr_db=test_snr,
        delta_f_normalized=0.005, phi0=np.pi / 6, rng=RNG
    )

    rx_bits_all, rx_symbols_rec = rx_pipeline(
        rx_channel, pulse, scheme, sps, len(symbols_tx), pilot_symbols=pilot_symbols, do_carrier_recovery=True
    )

    k = bits_per_symbol(scheme)
    rx_data_bits = rx_bits_all[N_PILOT * k :]

    min_len = min(len(tx_bits), len(rx_data_bits))
    bit_errors = np.sum(tx_bits[:min_len] != rx_data_bits[:min_len])
    ber = bit_errors / min_len
    print(f"[+] Bit analysis: {bit_errors} errors in {min_len} bits | BER = {ber:.6f}")

    audio_rec = bitstream_to_audio(rx_data_bits[:min_len], B=8)
    out_audio_path = os.path.join(out_dir, "reconstructed_phase2.wav")
    audio_rec_int16 = np.int16(audio_rec * 32767)
    wavfile.write(out_audio_path, fs, audio_rec_int16)
    print(f"[+] Reconstructed audio file '{out_audio_path}' saved")

    print("\n[+] Saving outputs ..")
    plot_constellation(scheme, os.path.join(out_dir, f"constellation_{scheme}.png"), rx_symbols_rec[N_PILOT:N_PILOT+2000])

    eye_data = build_eye_diagram_data(scheme, pulse_kind, sps=sps, r=0.35, span=12, snr_db=test_snr)
    plot_eye_diagram(eye_data, sps, os.path.join(out_dir, "eye_diagram.png"), f"Eye Diagram - {scheme.upper()} ({pulse_kind.upper()})")

    snr_range = np.arange(0, 22, 3)
    ber_no, ber_with = run_ber_simulation(scheme, pulse_kind, snr_range, sps=sps, n_data_symbols=4000, n_pilot=N_PILOT)
    plot_ber_curves(scheme, snr_range, ber_no, ber_with, os.path.join(out_dir, "ber_curves.png"))

    plot_adaptive_throughput(np.arange(0, 26, 1), os.path.join(out_dir, "adaptive_throughput.png"))

    print(f"\n[+] All done, all outputs saved in the '{out_dir}' folder.")
    print("=" * 60)

if __name__ == "__main__":
    run_full_phase2_pipeline()