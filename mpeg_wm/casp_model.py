"""Revised CASP auditory model (Paulick, Relano-Iborra & Dau, JASA 157:3232, 2025),
written from the paper (and the papers it cites for stages it does not print).

Pipeline, in the order sound travels through the ear:
  1. to_pascals              digital samples -> sound pressure          (this step)
  2. erb_channels            the 50 listening channels, 250 Hz - 8 kHz
  3. outer_middle_ear        ear canal + middle-ear bones
  4-7. drnl                  nonlinear cochlea (basilar membrane)
  8. inner_hair_cell         saturating transduction (new in the 2025 revision)
  9. adaptation              5 feedback loops -> forward masking
  10. modulation_filterbank  envelope split by wobble rate
  11. internal_representation, 12. dprime, 13. check_calibration
"""

import numpy as np

try:                                   # the two time loops (hair cell, adaptation) are
    from numba import njit             # sample-by-sample recursions; numba makes them
    HAVE_NUMBA = True                  # ~100x faster. Pure numpy is the fallback.
except Exception:                      # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        return (lambda f: f) if not args or not callable(args[0]) else args[0]

P_REF = 20e-6           # Pa: 0 dB SPL, the quietest pressure a young ear can hear
FULL_SCALE_DB = 96.0     # same convention as psychoacoustic.py: full-scale sine = 96 dB SPL


def to_pascals(x: np.ndarray, full_scale_db: float = FULL_SCALE_DB) -> np.ndarray:
    """Convert digital samples in [-1, 1] to sound pressure in Pascals.

    A full-scale sine (amplitude 1.0) comes out at `full_scale_db` dB SPL, the same
    playback-level assumption the MPEG model uses, so both models hear one loudness.
    """
    # SPL is defined on RMS pressure: L = 20*log10(p_rms / P_REF).
    # A sine of amplitude A has RMS A/sqrt(2), so for A = 1 to read full_scale_db:
    #   p_rms = P_REF * 10**(full_scale_db/20)  ->  peak = p_rms * sqrt(2).
    pascals_per_unit = P_REF * 10 ** (full_scale_db / 20) * np.sqrt(2)
    return np.asarray(x, dtype=np.float64) * pascals_per_unit


# --- ERB scale (Glasberg & Moore 1990) --------------------------------------------
# ERB(f) = 24.7 * (0.00437 f + 1) Hz is the width of the ear's filter at f.
# The ERB-rate (how many ERBs fit below f) is the integral of 1/ERB(f):
#   E(f) = 1 / (24.7 * 0.00437) * ln(1 + 0.00437 f) = 9.2645 * ln(1 + 0.00437 f)
ERB_K = 0.00437
ERB_C = 1.0 / (24.7 * ERB_K)          # 9.2645


def hz_to_erb_rate(f_hz):
    """Frequency (Hz) -> ERB-rate (number of ERBs below that frequency)."""
    return ERB_C * np.log1p(ERB_K * np.asarray(f_hz, dtype=float))


def erb_rate_to_hz(e):
    """ERB-rate -> frequency (Hz); the inverse of hz_to_erb_rate."""
    return np.expm1(np.asarray(e, dtype=float) / ERB_C) / ERB_K


def erb_channels(f_low: float = 250.0, f_high: float = 8000.0,
                 per_erb: float = 2.0) -> np.ndarray:
    """Centre frequencies of the ear's listening channels, equally spaced in ERB.

    Starts at f_low and steps 1/per_erb ERB until f_high (the paper: 250 Hz - 8 kHz,
    two channels per ERB). Returns Hz, low -> high.
    """
    e_low, e_high = hz_to_erb_rate(f_low), hz_to_erb_rate(f_high)
    n = int(np.floor((e_high - e_low) * per_erb + 1e-9)) + 1
    return erb_rate_to_hz(e_low + np.arange(n) / per_erb)


# --- Outer + middle ear ---------------------------------------------------------
# The paper (Sec. II B 1) uses the pair of linear-phase FIR filters of
# Lopez-Poveda & Meddis (2001), as in Jepsen et al. (2008). Measured data points:

# Headphone + outer ear gain (linear), Pralong & Carlile (1996) Fig. 1(e):
# Sennheiser HD 250 to eardrum. Peaks near 2.7 kHz (ear-canal resonance).
OUTER_EAR_HZ = np.array([125, 250, 500, 1000, 1237.4, 1531.1, 1894.6, 2002.5, 2344.3,
                         2721.3, 3001.4, 3589.5, 4001.3, 4441.5, 5004.2, 5495.9, 5997.4,
                         6800.5, 6946.9, 7995.5, 8414.9, 9008.4])
OUTER_EAR_GAIN = np.array([1.0, 1.0, 1.0, 0.99485, 0.99485, 0.99485, 1.11451, 1.23574,
                           1.86767, 2.82275, 2.18054, 1.44276, 1.17356, 1.37016, 1.59969,
                           1.37016, 1.11451, 0.64813, 0.63161, 0.27651, 0.08434, 0.08434])

# Middle ear: peak stapes velocity (m/s) for a 0 dB SPL (20 uPa) tone, Goode et al.
# (1994) as used by Lopez-Poveda & Meddis (2001) Fig. 2b. Peaks near 800 Hz.
MIDDLE_EAR_HZ = np.array([100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                          2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500,
                          6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
MIDDLE_EAR_VEL = np.array([1.181e-9, 2.363e-9, 4.728e-9, 7.577e-9, 1.000e-8, 8.235e-9,
                           6.240e-9, 5.585e-9, 5.000e-9, 4.232e-9, 3.787e-9, 3.000e-9,
                           2.715e-9, 2.498e-9, 2.174e-9, 1.893e-9, 1.742e-9, 1.516e-9,
                           1.117e-9, 1.320e-9, 1.214e-9, 9.726e-10, 9.460e-10, 8.705e-10,
                           8.000e-10, 7.577e-10, 7.168e-10, 6.781e-10, 6.240e-10, 6.000e-10])

EAR_FIR_TAPS = 513        # odd -> type-I linear phase; 16 kHz / 513 ~ 31 Hz resolution


def _linear_phase_fir(freqs_hz, gains, fs, numtaps=EAR_FIR_TAPS, gain_dc=None):
    """FIR whose magnitude passes through the measured (frequency, gain) points."""
    from scipy.signal import firwin2
    nyq = fs / 2
    keep = freqs_hz < nyq
    f = np.concatenate([[0.0], freqs_hz[keep], [nyq]])
    g = np.concatenate([[gains[0] if gain_dc is None else gain_dc], gains[keep],
                        [np.interp(nyq, freqs_hz, gains)]])
    return firwin2(numtaps, f, g, fs=fs)


def _fir_aligned(x, h):
    """Filter with FIR h and remove its (numtaps-1)/2 delay: output lines up with input."""
    from scipy.signal import fftconvolve
    d = (len(h) - 1) // 2
    return fftconvolve(x, h)[d:d + len(x)]


def outer_middle_ear(p: np.ndarray, fs: int) -> np.ndarray:
    """Eardrum pressure (Pa) -> peak stapes velocity (m/s).

    Outer ear: headphone-to-eardrum gain (unitless). Middle ear: stapes velocity per
    Pascal, i.e. the measured velocity at 0 dB SPL divided by 20 uPa.
    """
    h_outer = _linear_phase_fir(OUTER_EAR_HZ, OUTER_EAR_GAIN, fs)
    # Velocity grows ~proportionally to f below 800 Hz, so it goes to 0 at DC.
    h_middle = _linear_phase_fir(MIDDLE_EAR_HZ, MIDDLE_EAR_VEL / P_REF, fs, gain_dc=0.0)
    return _fir_aligned(_fir_aligned(np.asarray(p, dtype=np.float64), h_outer), h_middle)


# --- Gammatone filter (the cochlea's building block) ----------------------------

def gammatone(x: np.ndarray, fc: float, bw: float, fs: int, order: int) -> np.ndarray:
    """One gammatone band-pass filter of the given order, centred at fc (Hz).

    "Classic" all-pole form used by the DRNL implementations (AMT): one complex
    pole repeated `order` times,
        H(z) = (1 - e^-phi)^order / (1 - e^(-phi - j*theta) z^-1)^order,
        phi = 2*pi*bw/fs (how fast the ringing decays), theta = 2*pi*fc/fs.
    Filtering a real signal and keeping the real part gives a real band-pass whose
    peak gain at fc is 0.5. That convention is kept on purpose: the paper's DRNL
    gains were fitted with it.
    """
    from scipy.signal import lfilter
    phi = 2 * np.pi * bw / fs
    theta = 2 * np.pi * fc / fs
    pole = np.exp(-phi - 1j * theta)
    b = [(1 - np.exp(-phi)) ** order]
    a = np.poly(np.full(order, pole))            # (1 - pole z^-1)^order
    return np.real(lfilter(b, a, np.asarray(x, dtype=np.complex128)))


# --- DRNL parameters (paper, Table I, revised CASP column) -----------------------
# Every parameter follows log10(parameter) = p0 + m * log10(CF):  (p0, m)
DRNL_TABLE = {
    # linear path
    "lin_gain":   (4.20405, -0.47909),
    "lin_bw":     (0.03728, 0.75),       # gammatone bandwidth, Hz
    "lin_cf":     (-0.06762, 1.01),      # gammatone centre frequency, Hz
    "lin_lp":     (-0.06762, 1.017),     # low-pass cut-off, Hz
    # non-linear path
    "nlin_a":     (1.40298, 0.81916),    # broken stick: linear gain (low levels)
    "nlin_b":     (1.61912, -0.81867),   # broken stick: compressive gain (high levels)
    "nlin_c":     (-0.60206, 0.0),       # broken stick: exponent, 10**-0.60206 = 0.25
    "nlin_bw":    (-0.03193, 0.77),
    "nlin_cf":    (-0.05252, 1.0165),
    "nlin_lp":    (-0.05252, 1.0165),
}
DRNL_ORDERS = dict(lin_gt=2, lin_lp=4, nlin_gt=2, nlin_lp=1)   # cascade counts (bold in Table I)
COMPRESS_LIMIT_HZ = 1500.0   # a, b held constant above 1.5 kHz (Jepsen et al. 2008 / AMT drnl)


def drnl_params(cf) -> dict:
    """Per-channel DRNL settings for channel centre frequencies `cf` (Hz).

    Returns a dict of arrays (one value per channel) with the keys of DRNL_TABLE.
    """
    cf = np.asarray(cf, dtype=float)
    out = {}
    for name, (p0, m) in DRNL_TABLE.items():
        # The compressor's a and b stop changing above 1.5 kHz: measured basilar-
        # membrane compression is about the same at all high CFs.
        f = np.minimum(cf, COMPRESS_LIMIT_HZ) if name in ("nlin_a", "nlin_b") else cf
        out[name] = 10.0 ** (p0 + m * np.log10(f))
    return out


def broken_stick(x: np.ndarray, a, b, c) -> np.ndarray:
    """Cochlear compression: y = sign(x) * min(a*|x|, b*|x|**c).

    Quiet input follows the straight line a*|x| (high gain); loud input follows
    b*|x|**c with c = 0.25, so 10x more input gives only ~1.8x more output.
    x: (channels, time) or (time,); a, b, c: scalars or one value per channel.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:                       # per-channel parameters -> column vectors
        a, b, c = (np.asarray(v, dtype=float).reshape(-1, 1) for v in (a, b, c))
    mag = np.abs(x)
    return np.sign(x) * np.minimum(a * mag, b * mag ** c)


def _butter_lowpass(x, cutoff, fs, times):
    """2nd-order Butterworth low-pass, applied `times` times in a row."""
    from scipy.signal import butter, lfilter
    b, a = butter(2, min(cutoff, 0.999 * fs / 2), fs=fs)
    for _ in range(times):
        x = lfilter(b, a, x)
    return x


def drnl(v: np.ndarray, fs: int, cf=None) -> np.ndarray:
    """Dual-resonance nonlinear filterbank: stapes velocity -> basilar-membrane
    velocity in every channel.

    Per channel (Lopez-Poveda & Meddis 2001; orders and parameters from Table I):
      linear path:      gain -> gammatone (order 2) -> low-pass x4
      non-linear path:  gammatone (order 2) -> broken stick -> gammatone (order 2) -> low-pass x1
      output = linear + non-linear
    Returns an array of shape (channels, time).
    """
    cf = erb_channels() if cf is None else np.asarray(cf, dtype=float)
    P, n = drnl_params(cf), DRNL_ORDERS
    v = np.asarray(v, dtype=float)
    out = np.empty((len(cf), len(v)))
    for i in range(len(cf)):
        lin = gammatone(P["lin_gain"][i] * v, P["lin_cf"][i], P["lin_bw"][i], fs, n["lin_gt"])
        lin = _butter_lowpass(lin, P["lin_lp"][i], fs, n["lin_lp"])

        nl = gammatone(v, P["nlin_cf"][i], P["nlin_bw"][i], fs, n["nlin_gt"])
        nl = broken_stick(nl, P["nlin_a"][i], P["nlin_b"][i], P["nlin_c"][i])
        nl = gammatone(nl, P["nlin_cf"][i], P["nlin_bw"][i], fs, n["nlin_gt"])
        nl = _butter_lowpass(nl, P["nlin_lp"][i], fs, n["nlin_lp"])

        out[i] = lin + nl
    return out


# --- Inner hair cell (new in the 2025 revision; Table I, SI units) --------------
IHC_COUPLING = 5.62e-6     # BM velocity -> stereocilia displacement (m)
IHC_GMAX = 30e-9           # S, all transduction channels open
IHC_U0, IHC_S0, IHC_S1 = 20e-9, 16e-9, 35e-9      # m, Boltzmann shape
IHC_CM = 12.5e-12          # F, membrane capacitance
IHC_EP = 90e-3             # V, endocochlear potential (the "battery" above the cell)
IHC_GKF = IHC_GKS = 19.8e-9                      # S, fast / slow potassium conductance
# Table I prints 71 and 78 mV without a sign; potassium reversal potentials are
# negative (with +71/+78 mV the cell would rest near +80 mV, which is not physical).
IHC_EKF, IHC_EKS = -71e-3, -78e-3


def met_conductance(u: np.ndarray) -> np.ndarray:
    """Eq. (1): opening of the transduction channels vs stereocilia displacement u (m).

    G = Gmax / [1 + exp(-(u-u0)/s0) * (1 + exp(-(u-u0)/s1))], an S-curve that is
    partly open at rest, opens further for positive u and saturates at Gmax.
    """
    e0 = np.exp(-(u - IHC_U0) / IHC_S0)
    e1 = np.exp(-(u - IHC_U0) / IHC_S1)
    return IHC_GMAX / (1.0 + e0 * (1.0 + e1))


def inner_hair_cell(bm: np.ndarray, fs: int) -> np.ndarray:
    """Basilar-membrane velocity (channels, time) -> receptor potential (V) relative
    to rest, same shape.

    Eq. (2):  -Cm dV/dt = G_MET(u)(V - EP) + G_Kf(V - E_Kf) + G_Ks(V - E_Ks).
    Integrated exactly for a conductance held constant over each sample: the cell
    relaxes towards V_inf = sum(G_i E_i) / sum(G_i) with time constant Cm / sum(G_i).
    """
    g_met = met_conductance(IHC_COUPLING * np.asarray(bm, dtype=float))
    g_k = IHC_GKF + IHC_GKS
    g_tot = g_met + g_k
    v_inf = (g_met * IHC_EP + IHC_GKF * IHC_EKF + IHC_GKS * IHC_EKS) / g_tot
    keep = np.exp(-g_tot / (IHC_CM * fs))           # fraction of the old voltage kept per sample

    # Start at the resting potential (silence: u = 0), so there is no start-up transient.
    g0 = met_conductance(np.array(0.0))
    v_rest = (g0 * IHC_EP + IHC_GKF * IHC_EKF + IHC_GKS * IHC_EKS) / (g0 + g_k)
    v = _relax(v_inf, keep, np.full(g_met.shape[0], float(v_rest)))
    return v - v_rest


@njit(cache=True)
def _relax(v_inf, keep, v0):
    """v[t] = v_inf[t] + (v[t-1] - v_inf[t]) * keep[t], per channel."""
    n_ch, n_t = v_inf.shape
    v = np.empty_like(v_inf)
    v_now = v0.copy()
    for t in range(n_t):
        for c in range(n_ch):
            v_now[c] = v_inf[c, t] + (v_now[c] - v_inf[c, t]) * keep[c, t]
            v[c, t] = v_now[c]
    return v


# --- Adaptation loops (Dau et al. 1996; revised constants, Table I) -------------
ADAPT_TAU = np.array([0.007, 0.032, 0.088, 0.214, 0.5])   # s, one per feedback loop
ADAPT_LIMIT = 10.0          # an onset may overshoot the steady response at most ~10x

# Hearing threshold in quiet, free field, ages 18-25 (ISO 389-7 / ISO 226), dB SPL.
# The paper sets each channel's minimum input level from these at the channel's CF.
ISO_THRESHOLD_HZ = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                             400, 500, 630, 750, 800, 1000, 1250, 1500, 1600, 2000, 2500,
                             3000, 3150, 4000, 5000, 6000, 6300, 8000, 9000, 10000, 11200,
                             12500, 14000, 16000, 18000])
ISO_THRESHOLD_DB = np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
                             14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.4, 2.2, 2.4, 3.5, 2.4, 1.7,
                             -1.3, -4.2, -5.8, -6.0, -5.4, -1.5, 4.3, 6.0, 12.6, 13.9, 13.9,
                             13.0, 12.3, 18.4, 40.2, 73.2])

_MIN_LEVEL_CACHE = {}


def adaptation_min_levels(cf, fs: int) -> np.ndarray:
    """Per-channel floor for the adaptation input: the hair-cell response (RMS, V)
    to a tone at the channel's CF played exactly at the threshold of hearing.

    Anything quieter than this is treated as silence, so the model's "threshold in
    quiet" matches people's (paper, Sec. II B 3).
    """
    cf = np.asarray(cf, dtype=float)
    key = (fs, tuple(np.round(cf, 3)))
    if key not in _MIN_LEVEL_CACHE:
        t = np.arange(int(0.2 * fs)) / fs
        levels = np.empty(len(cf))
        for i, f in enumerate(cf):
            thr_db = np.interp(np.log(f), np.log(ISO_THRESHOLD_HZ), ISO_THRESHOLD_DB)
            tone = to_pascals(10 ** ((thr_db - FULL_SCALE_DB) / 20) * np.sin(2 * np.pi * f * t))
            v = inner_hair_cell(drnl(outer_middle_ear(tone, fs), fs, [f]), fs)[0]
            levels[i] = np.sqrt(np.mean(v[len(v) // 2:] ** 2))       # steady second half
        _MIN_LEVEL_CACHE[key] = levels
    return _MIN_LEVEL_CACHE[key]


def adaptation(x: np.ndarray, fs: int, min_level: np.ndarray,
               tau=ADAPT_TAU, limit: float = ADAPT_LIMIT) -> np.ndarray:
    """Five divisive feedback loops in series (Puschel 1988; Dau et al. 1996).

    Each loop divides its input by a slowly low-passed copy of its own output. For a
    steady input x this settles at x**(1/2) per loop, so five loops give x**(1/32):
    roughly logarithmic, like loudness. Sudden changes pass before the loops catch up
    (onset emphasis); after a loud sound stops the loops are still dividing by a large
    number, so the next few hundred ms are pushed down: forward masking.

    x: hair-cell output (channels, time), V. min_level: one floor per channel.
    Returns model units (MU): 0 for a signal at threshold, larger for louder.
    """
    x = np.maximum(np.asarray(x, dtype=float), min_level[:, None])
    keep = np.exp(-1.0 / (np.asarray(tau) * fs))           # low-pass memory per loop
    # Steady state at the floor: state_1 = sqrt(min), state_k = sqrt(state_{k-1}).
    state = np.empty((len(min_level), len(tau)))
    state[:, 0] = np.sqrt(min_level)
    for k in range(1, len(tau)):
        state[:, k] = np.sqrt(state[:, k - 1])
    # Overshoot limit: above 1 the loop output is bent smoothly towards a ceiling of
    # about limit * (1 - state^2), with value 1 and slope 1 where the bend starts.
    peak = (1.0 - state ** 2) * limit - 1.0
    expfac = -2.0 / peak
    offset = peak - 1.0

    out = _adapt_loops(x, state, keep, peak, expfac, offset, float(limit))
    # Model units: 0 at the floor, 100 for a steady input of 1.
    corr = min_level ** (1.0 / 2 ** len(tau))
    return (out - corr[:, None]) * (100.0 / (1.0 - corr))[:, None]


@njit(cache=True)
def _adapt_loops(x, state, keep, peak, expfac, offset, limit):
    """Sample-by-sample loop core of `adaptation` (state is updated in place)."""
    n_ch, n_t = x.shape
    n_loops = state.shape[1]
    out = np.empty_like(x)
    for t in range(n_t):
        for c in range(n_ch):
            y = x[c, t]
            for k in range(n_loops):
                y = y / state[c, k]                                   # divide by memory
                if limit > 1.0 and y > 1.0:                           # overshoot limit
                    y = 2.0 * peak[c, k] / (1.0 + np.exp(expfac[c, k] * (y - 1.0))) - offset[c, k]
                state[c, k] = keep[k] * state[c, k] + (1.0 - keep[k]) * y   # memory follows output
            out[c, t] = y
    return out


# --- Modulation filterbank (Dau et al. 1997; Table I) ---------------------------
MOD_PRE_LP_HZ = 150.0      # 1st-order low-pass before the filterbank (Ewert & Dau 2000)
MOD_DC_LP_HZ = 2.5         # the lowest "filter" is a 2.5 Hz low-pass (very slow changes)
MOD_Q = 2.0                # above 10 Hz: bandwidth = centre / Q
MOD_MAX_HZ = 1000.0        # paper: filters span 2.5 - 1000 Hz
MOD_LOW_BW_HZ = 5.0        # at and below 10 Hz: constant 5 Hz bandwidth
IR_DECIMATE = 4            # envelopes are slow: 16 kHz -> 4 kHz internal representation


def modulation_centres(cf_hz: float) -> list:
    """Modulation filter centres for one auditory channel; 0 means the 2.5 Hz low-pass.

    5 and 10 Hz (5 Hz wide), then steps of (1 + 1/2Q)/(1 - 1/2Q) = 5/3 so neighbouring
    Q = 2 filters touch, up to min(CF/4, 1000 Hz): a channel cannot carry envelope
    fluctuations faster than about a quarter of its own frequency.
    """
    top = min(cf_hz / 4.0, MOD_MAX_HZ)
    centres = [0.0] + [f for f in (5.0, 10.0) if f <= top]
    ratio = (1 + 1 / (2 * MOD_Q)) / (1 - 1 / (2 * MOD_Q))
    f = 10.0 * ratio
    while f <= top:
        centres.append(f)
        f *= ratio
    return centres


def modulation_filterbank(x: np.ndarray, fs: int, cf) -> tuple:
    """Split each channel's adapted envelope by how fast it fluctuates.

    x: adaptation output (channels, time) at fs. Returns (ir, mfc, fs_ir):
      ir   (channels, n_mod, time_ir), zeros where a channel has fewer filters
      mfc  (channels, n_mod) centre frequencies, NaN where absent
      fs_ir = fs / IR_DECIMATE
    Below 10 Hz the filter's real part is kept (envelope phase matters for slow
    changes); at and above 10 Hz its magnitude / sqrt(2) (phase is not heard).
    """
    from scipy.signal import butter, lfilter, resample_poly
    b, a = butter(1, MOD_PRE_LP_HZ, fs=fs)
    env = resample_poly(lfilter(b, a, np.asarray(x, dtype=float), axis=1), 1, IR_DECIMATE, axis=1)
    fs_ir = fs / IR_DECIMATE
    centres = [modulation_centres(f) for f in cf]
    n_mod = max(len(c) for c in centres)
    ir = np.zeros((len(cf), n_mod, env.shape[1]))
    mfc = np.full((len(cf), n_mod), np.nan)
    b_dc, a_dc = butter(2, MOD_DC_LP_HZ, fs=fs_ir)
    for i, cs in enumerate(centres):
        for j, f in enumerate(cs):
            mfc[i, j] = f
            if f == 0.0:
                ir[i, j] = lfilter(b_dc, a_dc, env[i])
                continue
            # Complex one-pole resonator at f with -3 dB bandwidth bw, unity gain at f.
            bw = MOD_LOW_BW_HZ if f <= 10.0 else f / MOD_Q
            e0 = np.exp(-np.pi * bw / fs_ir)
            y = lfilter([1 - e0], [1, -e0 * np.exp(2j * np.pi * f / fs_ir)], env[i].astype(complex))
            ir[i, j] = np.real(y) if f < 10.0 else np.abs(y) / np.sqrt(2)
    return ir, mfc, fs_ir


# --- The whole model and its decision stage ---------------------------------------

def internal_representation(x: np.ndarray, fs: int, cf=None) -> dict:
    """Digital audio (mono, [-1, 1]) -> CASP internal representation.

    Chains functions 1-10. Returns dict(ir=(channels, n_mod, time_ir), mfc, fs_ir, cf).
    """
    cf = erb_channels() if cf is None else np.asarray(cf, dtype=float)
    v = inner_hair_cell(drnl(outer_middle_ear(to_pascals(x), fs), fs, cf), fs)
    adapted = adaptation(v, fs, adaptation_min_levels(cf, fs))
    ir, mfc, fs_ir = modulation_filterbank(adapted, fs, cf)
    return dict(ir=ir, mfc=mfc, fs_ir=fs_ir, cf=cf)


def ir_distance(ir_a: np.ndarray, ir_b: np.ndarray, fs_ir: float) -> float:
    """sqrt( (1/fs_ir) * sum (IR_a - IR_b)^2 ): how far apart two internal pictures are.

    With the template equal to the difference itself and constant internal noise,
    the ideal observer's d' is this distance divided by the noise (paper Eq. 3; the
    1/fs makes it independent of the IR sample rate).
    """
    return float(np.sqrt(np.sum((ir_a - ir_b) ** 2) / fs_ir))


JND_DB = 1.0          # assumed intensity JND for the calibration tone (paper's fitting condition)
_D_JND = {}


def calibration_distance(fs: int = 16000, level_db: float = 60.0, jnd_db: float = JND_DB) -> float:
    """IR distance of a just-noticeable change: a 1 kHz, 200 ms tone (10 ms ramps) at
    `level_db` vs `level_db + jnd_db`. This is "one unit of audibility" (d' = 1)."""
    key = (fs, level_db, jnd_db)
    if key not in _D_JND:
        n = int(0.2 * fs)
        t = np.arange(n) / fs
        ramp = np.ones(n)
        r = int(0.01 * fs)
        ramp[:r] = ramp[-r:][::-1] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
        tone = np.concatenate([np.zeros(int(0.1 * fs)), np.sin(2 * np.pi * 1000 * t) * ramp,
                               np.zeros(int(0.1 * fs))])
        amp = 10 ** ((level_db - FULL_SCALE_DB) / 20)
        a = internal_representation(amp * tone, fs)
        b = internal_representation(amp * 10 ** (jnd_db / 20) * tone, fs)
        _D_JND[key] = ir_distance(a["ir"], b["ir"], a["fs_ir"])
    return _D_JND[key]


def dprime(ir_a: np.ndarray, ir_b: np.ndarray, fs_ir: float, fs: int = 16000) -> float:
    """Audibility of the difference between two internal pictures, in JND units."""
    return ir_distance(ir_a, ir_b, fs_ir) / calibration_distance(fs)


def check_calibration(fs: int = 16000) -> dict:
    """Sanity checks against the paper's fitting condition (printed by test_casp.py).

    Weber's law: a 1 dB step should be about equally audible at 40, 60 and 80 dB, and
    a 2 dB step about twice as audible as a 1 dB step.
    """
    d60 = calibration_distance(fs, 60.0, 1.0)
    return dict(d_jnd=d60,
                two_db=calibration_distance(fs, 60.0, 2.0) / d60,
                at_40db=calibration_distance(fs, 40.0, 1.0) / d60,
                at_80db=calibration_distance(fs, 80.0, 1.0) / d60)
