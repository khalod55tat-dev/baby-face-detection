"""
AM + FM (Transmitter -> AWGN Channel -> Receiver) Simulation in Python
Outputs:
- Time-domain plots (message, AM/FM TX, RX, recovered)
- Frequency spectra (FFT magnitude)
- Key calculations (AM modulation index & efficiency, AM BW; FM beta, Carson BW)
Matches the style of your guide report and your course slides.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Optional SciPy (nice filters + Hilbert). If not available, code still runs with simple filters.
try:
    from scipy.signal import hilbert, firwin, lfilter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =========================
# Helper functions
# =========================
def awgn(x, snr_db, seed=0):
    """Add AWGN to signal x to reach target SNR (dB) based on signal power."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(x**2)
    snr_lin = 10 ** (snr_db / 10.0)
    p_noise = p_sig / snr_lin
    n = rng.normal(0, np.sqrt(p_noise), size=x.shape)
    return x + n

def fft_mag_db(x, fs):
    """FFT magnitude in dB (with fftshift)"""
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x * np.hanning(N)))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    mag = 20*np.log10(np.maximum(np.abs(X)/N, 1e-12))
    return f, mag

def lowpass(x, fs, cutoff_hz):
    """
    Low-pass filter.
    - If SciPy exists: FIR + lfilter
    - Else: simple moving average (rough but works)
    """
    if cutoff_hz <= 0:
        return x

    if SCIPY_OK:
        numtaps = 401  # FIR length
        b = firwin(numtaps, cutoff_hz/(fs/2))
        return lfilter(b, 1.0, x)
    else:
        # moving-average window ~ equivalent cutoff (very rough)
        win = int(max(5, fs/(cutoff_hz*10)))
        win = min(win, max(5, len(x)//50))
        w = np.ones(win)/win
        return np.convolve(x, w, mode="same")

def plot_time(t, signals, title, xlim=None):
    plt.figure()
    for label, sig in signals:
        plt.plot(t, sig, label=label)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.legend()

def plot_spectrum(x, fs, title, xlim=None):
    f, mag = fft_mag_db(x, fs)
    plt.figure()
    plt.plot(f, mag)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)


# =========================
# Simulation parameters
# =========================
fs = 200_000           # sampling frequency (Hz)
T  = 0.20              # duration (s)
t  = np.arange(0, T, 1/fs)

SNR_dB = 10            # channel SNR (dB) - change to see effect of noise

# Carrier & message frequencies like the guide report (scaled-down carrier for simulation)
fc_am = 10_000         # AM carrier (Hz)
fc_fm = 10_000         # FM carrier (Hz)

# AM message: two-tone (100 Hz and 200 Hz) like the guide report style
fm1, fm2 = 100, 200
Em1, Em2 = 2.0, 1.0    # message amplitudes (V)
Ec_am = 10.0           # AM carrier amplitude (V)

vm_am = Em1*np.cos(2*np.pi*fm1*t) + Em2*np.cos(2*np.pi*fm2*t)   # message

# FM message: single tone (100 Hz), beta = 2 -> Δf = beta*fm = 200 Hz
fm_fm  = 100
Ec_fm  = 15.0
beta   = 2.0
delta_f = beta * fm_fm      # frequency deviation (Hz)
vm_fm = 1.0*np.cos(2*np.pi*fm_fm*t)  # 1 V amplitude

# kf selection based on: f_i(t) = f_c + (kf/(2π))*v_m(t)
# => Δf = (kf/(2π)) * Vp  => kf = 2π Δf / Vp
Vp_fm = np.max(np.abs(vm_fm))
kf = 2*np.pi * delta_f / Vp_fm     # rad/s/Volt


# =========================
# TRANSMITTER
# =========================
# --- AM DSB-FC (full carrier): vAM(t) = [Ec + vm(t)] cos(2π fc t)
s_am = (Ec_am + vm_am) * np.cos(2*np.pi*fc_am*t)

# --- FM: vFM(t) = Ec cos(2π fc t + kf ∫ vm(t) dt)
phi = 2*np.pi*fc_fm*t + kf * np.cumsum(vm_fm) * (1/fs)   # integrate via cumulative sum
s_fm = Ec_fm * np.cos(phi)

# =========================
# CHANNEL (AWGN)
# =========================
r_am = awgn(s_am, SNR_dB, seed=1)
r_fm = awgn(s_fm, SNR_dB, seed=2)

# =========================
# RECEIVER (DEMODULATION)
# =========================
# --- AM envelope detection (Hilbert) + lowpass (baseband recovery)
if SCIPY_OK:
    env_am = np.abs(hilbert(r_am))
else:
    # fallback: rectifier + lowpass (less accurate than Hilbert)
    env_am = np.abs(r_am)

rec_am = lowpass(env_am - Ec_am, fs, cutoff_hz=1_000)  # recover message (<= 200 Hz)

# --- FM demodulation via analytic signal phase derivative:
#     f_inst(t) = (1/(2π)) d/dt (unwrap(angle(analytic)))
if SCIPY_OK:
    z = hilbert(r_fm)
    # limiter effect (reduce amplitude noise impact): normalize magnitude
    z = z / np.maximum(np.abs(z), 1e-12)

    phase = np.unwrap(np.angle(z))
    dphase = np.diff(phase) * fs
    f_inst = dphase / (2*np.pi)

    # align length (diff reduces by 1)
    f_inst = np.concatenate([f_inst[:1], f_inst])

    # remove carrier and scale back to volts
    rec_fm_raw = (f_inst - fc_fm) * (Vp_fm / delta_f)

    rec_fm = lowpass(rec_fm_raw, fs, cutoff_hz=1_000)
else:
    # fallback (no SciPy): very rough FM recovery not provided
    rec_fm = np.zeros_like(t)

# =========================
# Theory calculations (for your report discussion)
# =========================
# AM:
mu1 = Em1 / Ec_am
mu2 = Em2 / Ec_am
mu_total_rss = np.sqrt(mu1**2 + mu2**2)  # combined modulation index (RMS-style for multi-tone)

# For multi-tone AM efficiency:
# sideband power ratio = (mu1^2 + mu2^2)/2 ; total ratio = 1 + (mu1^2 + mu2^2)/2
eta_am = ((mu1**2 + mu2**2)/2) / (1 + (mu1**2 + mu2**2)/2) * 100.0

fm_max = max(fm1, fm2)
bw_am_theory = 2 * fm_max  # DSB bandwidth

# FM (Carson's rule):
bw_fm_carson = 2 * (delta_f + fm_fm)  # = 2(β+1)fm

print("===== AM (DSB-FC) =====")
print(f"Carrier amplitude Ec = {Ec_am:.2f} V, fc = {fc_am} Hz")
print(f"Message tones: {Em1}cos(2π{fm1}t) + {Em2}cos(2π{fm2}t)")
print(f"Modulation indices: mu1={mu1:.3f}, mu2={mu2:.3f}, combined(mu_rss)={mu_total_rss:.3f}")
print(f"Theoretical AM bandwidth (DSB): BW = 2*fmax = {bw_am_theory} Hz")
print(f"AM power efficiency (multi-tone): η = {eta_am:.3f}%")

print("\n===== FM =====")
print(f"Carrier amplitude Ec = {Ec_fm:.2f} V, fc = {fc_fm} Hz")
print(f"fm = {fm_fm} Hz, beta = {beta:.2f} => Δf = {delta_f:.2f} Hz")
print(f"Carson bandwidth: BW = 2(Δf + fm) = {bw_fm_carson:.2f} Hz")
print(f"SNR(channel) = {SNR_dB} dB, SciPy available = {SCIPY_OK}")

# =========================
# PLOTS (Results section)
# =========================
# Zoom window to clearly see carrier waveforms
t_zoom = (0.00, 0.01)

# --- Time domain
plot_time(t, [("vm_am(t) (message)", vm_am)], "AM: Message signal vm_am(t)", xlim=t_zoom)
plot_time(t, [("s_am(t) TX", s_am)], "AM: Transmitted (DSB-FC) signal", xlim=t_zoom)
plot_time(t, [("r_am(t) RX", r_am)], f"AM: Received signal with AWGN (SNR={SNR_dB} dB)", xlim=t_zoom)
plot_time(t, [("Recovered AM", rec_am), ("Original vm_am", vm_am)], "AM: Demodulated (envelope + LPF)", xlim=t_zoom)

plot_time(t, [("vm_fm(t) (message)", vm_fm)], "FM: Message signal vm_fm(t)", xlim=t_zoom)
plot_time(t, [("s_fm(t) TX", s_fm)], "FM: Transmitted FM signal", xlim=t_zoom)
plot_time(t, [("r_fm(t) RX", r_fm)], f"FM: Received signal with AWGN (SNR={SNR_dB} dB)", xlim=t_zoom)
plot_time(t, [("Recovered FM", rec_fm), ("Original vm_fm", vm_fm)], "FM: Demodulated (phase-derivative + LPF)", xlim=t_zoom)

# --- Frequency spectra (full + zoom near carrier)
plot_spectrum(s_am, fs, "AM Spectrum (TX)", xlim=(fc_am-2000, fc_am+2000))
plot_spectrum(r_am, fs, "AM Spectrum (RX with noise)", xlim=(fc_am-2000, fc_am+2000))

plot_spectrum(s_fm, fs, "FM Spectrum (TX)", xlim=(fc_fm-3000, fc_fm+3000))
plot_spectrum(r_fm, fs, "FM Spectrum (RX with noise)", xlim=(fc_fm-3000, fc_fm+3000))

plt.show()
