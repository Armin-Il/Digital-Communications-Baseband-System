import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal as sig
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# --- Audio loading ---

def load_audio(path, target_fs=16000):
    fs, data = wavfile.read(path)

    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:  # float32/float64 already
        x = data.astype(np.float64)

    if x.ndim > 1:
        x = x.mean(axis=1)  # downmix to mono

    if fs != target_fs:
        n_new = int(round(len(x) * target_fs / fs))
        x = sig.resample(x, n_new)
        fs = target_fs

    peak = np.max(np.abs(x)) + 1e-12
    x = x / peak
    return fs, x


def save_wav(path, fs, x):
    x_clip = np.clip(x, -1.0, 1.0)
    x_int16 = np.int16(x_clip * 32767)
    wavfile.write(path, fs, x_int16)


def generate_synthetic_test_audio(fs=16000, seconds=3.0, seed=0):
    # Fallback: if no real voice file is found, generate a synthetic
    # speech-like test signal instead
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.7 * t)) ** 1.5
    formants = [300, 900, 2200]
    x = np.zeros_like(t)
    for f0 in formants:
        x += np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    x *= envelope
    x += 0.02 * rng.standard_normal(len(t))
    x = x / (np.max(np.abs(x)) + 1e-12)
    return fs, x


# --- Uniform quantization ---

def uniform_quantize(x, B):
    L = 2 ** B
    step = 2.0 / L
    idx = np.floor((x + 1.0) / step)
    idx = np.clip(idx, 0, L - 1).astype(np.int64)
    xq = -1.0 + step / 2.0 + idx * step
    return xq, idx, step


def compute_mse_sqnr(x, xq):
    e = x - xq
    Px = np.mean(x ** 2)
    Pe = np.mean(e ** 2)
    sqnr_db = 10 * np.log10(Px / Pe) if Pe > 0 else np.inf
    return Pe, sqnr_db, e


def plot_error_histogram(e, B, out_path, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(e, bins=bins, density=True, color="#3b6fa0", edgecolor="black", alpha=0.8)
    plt.title(f"Histogram of Uniform Quantization Error (B={B} bits)")
    plt.xlabel("e[n] = x[n] - x_hat[n]")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# --- Mu-Law quantization ---

def mu_law_compress(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def mu_law_expand(y, mu=255):
    return np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu


def mu_law_quantize(x, B, mu=255):
    y = mu_law_compress(x, mu)
    yq, idx, step = uniform_quantize(y, B)
    xq = mu_law_expand(yq, mu)
    return xq, idx, step


def verify_companding_invertible(mu=255, tol=1e-9):
    x_test = np.linspace(-1, 1, 10001)
    y = mu_law_compress(x_test, mu)
    x_back = mu_law_expand(y, mu)
    max_err = np.max(np.abs(x_test - x_back))
    ok = max_err < tol
    return ok, max_err


# --- Bitstream generation ---

def levels_to_bits(idx, B):
    idx = np.asarray(idx, dtype=np.uint32)
    shifts = np.arange(B - 1, -1, -1)
    bits = ((idx[:, None] >> shifts) & 1).astype(np.uint8)
    return bits.flatten()


def bits_to_levels(bits, B):
    n = len(bits) // B
    bits = np.asarray(bits[: n * B], dtype=np.uint32).reshape(n, B)
    weights = (1 << np.arange(B - 1, -1, -1)).astype(np.uint32)
    idx = bits.dot(weights)
    return idx


def levels_to_signal(idx, B):
    L = 2 ** B
    step = 2.0 / L
    return -1.0 + step / 2.0 + idx.astype(np.float64) * step


def verify_bitstream_invertible(idx, B):
    bits = levels_to_bits(idx, B)
    idx_back = bits_to_levels(bits, B)
    ok = np.array_equal(idx, idx_back)
    return ok, bits


# --- Line Coding ---

def polar_nrz(bits, A=1.0, sps=8):
    symbols = np.where(bits == 1, A, -A).astype(np.float64)
    return np.repeat(symbols, sps)


def polar_rz(bits, A=1.0, sps=8):
    half = sps // 2
    symbols = np.where(bits == 1, A, -A).astype(np.float64)
    wave = np.repeat(symbols, sps)
    mask = np.tile(np.r_[np.ones(half), np.zeros(sps - half)], len(bits))
    return wave * mask


def manchester(bits, A=1.0, sps=8):
    half = sps // 2
    pattern1 = np.r_[np.full(half, A), np.full(sps - half, -A)]
    patterns = np.where(bits[:, None] == 1, pattern1[None, :], -pattern1[None, :])
    return patterns.flatten().astype(np.float64)


LINE_CODERS = {"nrz": polar_nrz, "rz": polar_rz, "manchester": manchester}


def get_pulse_shape(code, sps, A=1.0):
    if code == "nrz":
        return np.full(sps, A, dtype=np.float64)
    if code == "rz":
        half = sps // 2
        return np.r_[np.full(half, A), np.zeros(sps - half)]
    if code == "manchester":
        half = sps // 2
        return np.r_[np.full(half, A), np.full(sps - half, -A)]
    raise ValueError(code)


def estimate_psd(wave, fs_sim, nperseg=4096):
    nperseg = min(nperseg, len(wave))
    f, Pxx = sig.welch(wave, fs=fs_sim, nperseg=nperseg, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    return f, Pxx


def bandwidth_3db(f, Pxx):
    Pxx_db = 10 * np.log10(Pxx / (np.max(Pxx) + 1e-30) + 1e-30)
    mask = Pxx_db >= -3
    if not np.any(mask):
        return 0.0
    f_pos = f[mask]
    return float(np.max(np.abs(f_pos)) * 2)


def has_dc_null(f, Pxx, tol_frac=0.02):
    idx0 = np.argmin(np.abs(f))
    return Pxx[idx0] < tol_frac * np.max(Pxx)


def plot_waveform(wave, fs_sim, title, out_path, n_bits_show=20, sps=8):
    n_show = n_bits_show * sps
    t = np.arange(min(n_show, len(wave))) / fs_sim
    plt.figure(figsize=(8, 3))
    plt.step(t * 1e3, wave[: len(t)], where="post")
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_psd(f, Pxx, title, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(f, 10 * np.log10(Pxx / np.max(Pxx) + 1e-30))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized PSD (dB)")
    plt.axhline(-3, color="r", linestyle="--", linewidth=1, label="-3dB")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# --- AWGN channel ---

def add_awgn(wave, sps, Tb, EbN0_dB, rng=RNG):
    dt = Tb / sps
    n_bits = len(wave) // sps
    seg = wave[: n_bits * sps].reshape(n_bits, sps)
    Eb = float(np.mean(np.sum(seg ** 2, axis=1) * dt))

    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    N0 = Eb / EbN0_lin
    sigma2 = N0 / (2.0 * dt)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=wave.shape)
    return wave + noise, N0, Eb


# --- BER ---

def matched_filter_detect(rx, code, sps, A=1.0):
    pulse = get_pulse_shape(code, sps, A)
    n_bits = len(rx) // sps
    seg = rx[: n_bits * sps].reshape(n_bits, sps)
    stat = seg @ pulse
    bits_hat = (stat > 0).astype(np.uint8)
    return bits_hat


def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


def theoretical_ber(EbN0_dB_array):
    ebn0_lin = 10 ** (np.asarray(EbN0_dB_array) / 10.0)
    return q_function(np.sqrt(2 * ebn0_lin))


def simulate_ber_curve(bits, code, EbN0_dB_list, sps=8, Tb=1.0, A=1.0, rng=RNG):
    tx = LINE_CODERS[code](bits, A, sps)
    ber_list = []
    for ebn0 in EbN0_dB_list:
        rx, N0, Eb = add_awgn(tx, sps, Tb, ebn0, rng=rng)
        bits_hat = matched_filter_detect(rx, code, sps, A)
        n = min(len(bits_hat), len(bits))
        errors = np.sum(bits_hat[:n] != bits[:n])
        ber_list.append(errors / n)
    return np.array(ber_list)


def plot_ber_curve(EbN0_dB_list, ber_sim, ber_theory, code, out_path):
    plt.figure(figsize=(7, 5))
    plt.semilogy(EbN0_dB_list, ber_sim, "o-", label="Simulation")
    plt.semilogy(EbN0_dB_list, ber_theory, "s--", label="Theoretical Q(sqrt(2Eb/N0))")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs Eb/N0 — {code.upper()}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# --- Phase 1: full end-to-end chain for one configuration ---

def full_chain(x, fs, B=8, use_mu_law=False, code="nrz", EbN0_dB=10.0,
               sps=8, out_prefix="chain", out_dir="."):
    if use_mu_law:
        xq, idx, step = mu_law_quantize(x, B)
    else:
        xq, idx, step = uniform_quantize(x, B)

    bits = levels_to_bits(idx, B)

    Tb = 1.0
    tx = LINE_CODERS[code](bits, 1.0, sps)

    rx, N0, Eb = add_awgn(tx, sps, Tb, EbN0_dB)

    bits_hat = matched_filter_detect(rx, code, sps, 1.0)
    n = min(len(bits_hat), len(bits))
    ber = np.mean(bits_hat[:n] != bits[:n])

    idx_hat = bits_to_levels(bits_hat[: (len(bits_hat) // B) * B], B)
    if use_mu_law:
        L = 2 ** B
        step_mu = 2.0 / L
        y_hat = -1.0 + step_mu / 2.0 + idx_hat.astype(np.float64) * step_mu
        x_hat = mu_law_expand(y_hat, 255)
    else:
        x_hat = levels_to_signal(idx_hat, B)

    n_common = min(len(x_hat), len(x))
    method_str = "mulaw" if use_mu_law else "uniform"
    out_path = os.path.join(out_dir, f"{out_prefix}_{method_str}_B{B}_{code}_EbN0_{EbN0_dB}dB.wav")
    save_wav(out_path, fs, x_hat[:n_common])

    return {
        "ber": ber,
        "out_wav": out_path,
        "n_bits": len(bits),
        "N0": N0,
        "Eb": Eb,
    }


# --- Main pipeline ---

def run_full_project(audio_path=None, out_dir="results", B_list=(8, 4, 2),
                      EbN0_dB_list=(-5, 0, 5, 10, 15, 20)):
    os.makedirs(out_dir, exist_ok=True)
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(str(s))

    # Step 1: load audio
    if audio_path and os.path.exists(audio_path):
        fs, x = load_audio(audio_path)
        log(f"[Step 1] Loaded audio file: {audio_path} | fs={fs}Hz | length={len(x)/fs:.2f}s")
    else:
        fs, x = generate_synthetic_test_audio()
        log("[Step 1] Warning: no real voice file found; using a synthetic "
            "test signal instead. Provide a real WAV file for meaningful results.")
    save_wav(os.path.join(out_dir, "00_input_normalized.wav"), fs, x)

    log("\n[Step 2] Uniform Quantization")
    uniform_results = {}
    for B in B_list:
        xq, idx, step = uniform_quantize(x, B)
        Pe, sqnr_db, e = compute_mse_sqnr(x, xq)
        plot_error_histogram(e, B, os.path.join(out_dir, f"hist_uniform_B{B}.png"))
        save_wav(os.path.join(out_dir, f"reconstructed_uniform_B{B}.wav"), fs, xq)
        uniform_results[B] = dict(idx=idx, MSE=Pe, SQNR_dB=sqnr_db)
        log(f"  B={B}: MSE={Pe:.3e} | SQNR={sqnr_db:.2f} dB")

    # Step 3: Mu-Law quantization
    log("\n[Step 3] Mu-Law Quantization (Companding)")
    ok_inv, max_err = verify_companding_invertible()
    log(f"  Invertibility test F^-1(F(x)) ≈ x: {'OK' if ok_inv else 'FAIL'} (max_err={max_err:.2e})")

    mulaw_results = {}
    log(f"\n  {'B':>3} | {'SQNR Uniform(dB)':>18} | {'SQNR mu-Law(dB)':>16} | {'Difference':>10}")
    for B in B_list:
        xq_mu, idx_mu, _ = mu_law_quantize(x, B)
        Pe_mu, sqnr_mu, _ = compute_mse_sqnr(x, xq_mu)
        save_wav(os.path.join(out_dir, f"reconstructed_mulaw_B{B}.wav"), fs, xq_mu)
        diff = sqnr_mu - uniform_results[B]["SQNR_dB"]
        mulaw_results[B] = dict(idx=idx_mu, MSE=Pe_mu, SQNR_dB=sqnr_mu)
        log(f"  {B:>3} | {uniform_results[B]['SQNR_dB']:>18.2f} | {sqnr_mu:>16.2f} | {diff:>10.2f}")

    # Step 4: select best quantizer & generate bitstream
    log("\n[Step 4] Selecting best quantizer & generating bitstream (B=8)")
    best_B = 8
    best_use_mu = mulaw_results[best_B]["SQNR_dB"] > uniform_results[best_B]["SQNR_dB"]
    best_idx = mulaw_results[best_B]["idx"] if best_use_mu else uniform_results[best_B]["idx"]
    ok_bits, bits = verify_bitstream_invertible(best_idx, best_B)
    log(f"  Selected method: {'Mu-Law' if best_use_mu else 'Uniform'} | B={best_B}")
    log(f"  Bitstream invertibility test: {'OK' if ok_bits else 'FAIL'}")
    log(f"  Total bits: {len(bits)} | Rb = B*Fs = {best_B*fs} bit/s")

    # Step 5: Line coding
    log("\n[Step 5] Line Coding (Polar NRZ / Polar RZ / Manchester)")
    sps = 8
    Tb_sim = 1.0
    fs_sim = sps / Tb_sim
    line_bw = {}
    bits_demo = bits[:2000]
    for code in ["nrz", "rz", "manchester"]:
        wave = LINE_CODERS[code](bits_demo, 1.0, sps)
        plot_waveform(wave, fs_sim, f"Waveform {code.upper()} (First 20 Bits)",
                      os.path.join(out_dir, f"waveform_{code}.png"), n_bits_show=20, sps=sps)
        f, Pxx = estimate_psd(wave, fs_sim)
        plot_psd(f, Pxx, f"Estimated PSD — {code.upper()}", os.path.join(out_dir, f"psd_{code}.png"))
        bw = bandwidth_3db(f, Pxx)
        dc_null = has_dc_null(f, Pxx)
        line_bw[code] = bw
        log(f"  {code.upper():<11}: Bandwidth (-3dB) ≈ {bw:.3f} x Rb | Spectral null at DC: {'yes' if dc_null else 'no'}")
    if line_bw.get("manchester", 0) > 0 and line_bw.get("rz", 1) > 0:
        log(f"  BW(Manchester)/BW(RZ) ≈ {line_bw['manchester']/line_bw['rz']:.2f}  (expected ≈ 2)")

    # Step 6-7: AWGN channel and BER
    log("\n[Step 6-7] AWGN Channel and BER vs. Eb/N0")
    bits_ber = bits[:100000] if len(bits) >= 100000 else bits
    for code in ["nrz", "rz", "manchester"]:
        ber_sim = simulate_ber_curve(bits_ber, code, EbN0_dB_list, sps=sps, Tb=Tb_sim)
        ber_th = theoretical_ber(EbN0_dB_list)
        plot_ber_curve(EbN0_dB_list, ber_sim, ber_th, code,
                        os.path.join(out_dir, f"ber_{code}.png"))
        log(f"  {code.upper()}:")
        for ebn0, bs, bt in zip(EbN0_dB_list, ber_sim, ber_th):
            log(f"     Eb/N0={ebn0:>4} dB | BER_sim={bs:.4e} | BER_theory={bt:.4e}")

    # Step 8: final end-to-end chain
    log("\n[Step 8] Final End-to-End Chain")
    for B in (8, 4, 2):
        for ebn0 in (0, 10, 20):
            for use_mu in (False, True):
                res = full_chain(x, fs, B=B, use_mu_law=use_mu, code="nrz",
                                  EbN0_dB=ebn0, sps=sps, out_dir=out_dir)
                method_name = "Mu-Law" if use_mu else "Uniform"
                log(f"  B={B} ({method_name}), Eb/N0={ebn0}dB -> BER={res['ber']:.4e} -> {res['out_wav']}")

    with open(os.path.join(out_dir, "log_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    log(f"\nAll plots and reconstructed audio files saved in: {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_arg = sys.argv[1]
    else:
        audio_arg = input("Enter your WAV file name (e.g., my_voice.wav): ").strip()
        if not audio_arg:
            audio_arg = None

    run_full_project(audio_path=audio_arg, out_dir="results")