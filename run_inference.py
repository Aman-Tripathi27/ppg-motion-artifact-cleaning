"""Motion-artifact suppression for wrist PPG using IMU adaptive noise cancellation.

Inference entry point: ``clean_ppg(samples, channel_names, sampling_rate)`` returns
a cleaned 1-D PPG array of exactly the input length on the original 256 Hz grid.

Command line:
    python run_inference.py <input_record> <output.(npy|csv)>

``<input_record>`` is a WFDB record base path (or its .hea/.dat path). The cleaned
PPG is written as a 1-D ``.npy`` array or a single-column ``.csv``.

Only the wrist PPG and wrist IMU channels (accelerometers and gyroscope) are used;
no other signal or record metadata is read.

Pipeline:
    1. select channels by name; bridge non-finite and saturated samples
    2. zero-phase band-pass PPG and IMU to the cardiac band
    3. adaptive noise cancellation: subtract the IMU-predicted motion artifact
       in blocks whose fit is validated out of sample
    4. residual-motion band-pass
    5. two-peak correction: one pulse per local pulse period
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, coherence, find_peaks, sosfiltfilt, stft, welch

# --- Channels ---------------------------------------------------------------------
EXPECTED_FS = 256.0
PPG_CHANNEL = "wrist_ppg"
LOW_ACCEL = [f"wrist_low_noise_accelerometer_{a}" for a in "xyz"]
WIDE_ACCEL = [f"wrist_wide_range_accelerometer_{a}" for a in "xyz"]
GYRO = [f"wrist_gyro_{a}" for a in "xyz"]
# Low-noise and wide-range accelerometers are ~0.998-correlated in band; keeping both
# is harmless under ridge regularisation and helps if one clips. Absent channels are
# skipped rather than fatal.
IMU_REFERENCE = LOW_ACCEL + WIDE_ACCEL + GYRO

# --- Filtering and repair ---------------------------------------------------------
FILTER_ORDER = 3
DES_BAND = (0.4, 4.0)       # cardiac band for the PPG and the IMU references
FINAL_BAND = (1.4, 4.0)     # output band: suppresses residual low-frequency motion
LOW_RATE_BAND = (0.5, 4.0)  # output band when a clean low-rate pulse ridge is present
SATURATION_MIN_S = 0.1      # identical PPG samples for this long are an ADC rail
EPS = 1e-12                 # guard against division by zero in power normalisation

# --- Adaptive noise cancellation --------------------------------------------------
TAPS = 16                   # FIR taps per IMU channel
BLOCK_S = 8.0               # block length (s); blocks overlap by half
RIDGE = 1e-3                # ridge strength, relative to trace(G) / dim
RIDGE_ABS = 1e-1            # absolute ridge floor for weak or degenerate references
REFERENCE_VAR_EPS = 1e-12   # below this variance an IMU channel carries no motion
CONFIDENCE_CHUNK_S = 1.0    # alternating chunks for the out-of-sample check
CONFIDENCE_MIN = 0.1        # minimum held-out prediction score to subtract a block
ARTIFACT_RATIO_MAX = 0.98   # reject fits that explain almost the whole PPG block
MOTION_COHERENCE_MIN = 0.2  # record-level gate: PPG-IMU coherence at the PPG ridge

# --- Dominant-ridge features (record level) ----------------------------------------
RIDGE_MIN_S = 8.0           # shorter records skip the ridge analysis
RIDGE_NPERSEG = 1024        # Welch / coherence segment length
RIDGE_FMIN = 0.65           # lowest frequency searched for the dominant ridge (Hz)
RIDGE_HALF_WIDTH_HZ = 0.30  # power within this distance counts as the ridge
RIDGE_MAX_HZ = 1.6          # a clean low-rate ridge lies below this frequency ...
RIDGE_POWER_MIN = 0.75      # ... holds at least this share of in-band power ...
RIDGE_COHERENCE_MAX = 0.4   # ... and is not coherent with any IMU axis

# --- Pulse tracking ----------------------------------------------------------------
STFT_WIN_S = 8.0            # STFT window
STFT_HOP_S = 1.0            # one pulse estimate per second
STFT_NFFT = 4096            # ~0.0625 Hz bins
PULSE_FMIN = 0.75           # search range of the pulse fundamental (Hz)
PULSE_FMAX = 3.4
PULSE_CLIP_HZ = (0.5, 4.0)  # safe range before converting the track to a period
MOTION_SUPPRESS = 1.0       # PPG power is scaled by (1 - k * normalised IMU power)
MOTION_FLOOR = 0.02         # ... but never below this fraction
PATH_STEP_HZ = 0.15         # scale of frame-to-frame frequency change
PATH_PENALTY = 4.0          # weight of the path-smoothness penalty

# --- Two-peak correction -----------------------------------------------------------
DOUBLE_PEAK_FRAC = 0.5      # peaks closer than this fraction of the period are a double
DOUBLE_PEAK_CAP_S = 0.34    # never merge peaks further apart than this (s)
TRACK_CONFIDENCE_MIN = 0.3  # only merge where the track holds this share of band power


# ============================================================================ helpers
def _bandpass(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass (no filter delay); 1-D or column-wise 2-D."""
    sos = butter(FILTER_ORDER, [low, high], btype="bandpass", fs=EXPECTED_FS, output="sos")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return _filtfilt_safe(sos, x)
    return np.column_stack([_filtfilt_safe(sos, x[:, j]) for j in range(x.shape[1])])


def _filtfilt_safe(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """sosfiltfilt with reflect padding so very short inputs do not raise."""
    if x.size == 0:
        return x
    if x.size == 1:
        return np.zeros(1, dtype=np.float64)
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.size <= padlen:
        extra = padlen + 1 - x.size
        out = sosfiltfilt(sos, np.pad(x, (extra, extra), mode="reflect"))
        return np.asarray(out[extra:-extra], dtype=np.float64)
    return np.asarray(sosfiltfilt(sos, x), dtype=np.float64)


def _mark_saturation(x: np.ndarray) -> np.ndarray:
    """Mark PPG runs stuck at one value (ADC rails) as NaN so _repair bridges them.

    A real pulse never holds an identical sample for SATURATION_MIN_S; left in place,
    the flat run's edges would ring through the band-pass as false pulses.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    min_run = max(2, int(SATURATION_MIN_S * EXPECTED_FS))
    if x.size < min_run:
        return x
    change = np.flatnonzero(np.diff(x) != 0) + 1  # NaN != 0, so NaNs also end a run
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [x.size]])
    long_run = stops - starts >= min_run
    for start, stop in zip(starts[long_run], stops[long_run]):
        x[start:stop] = np.nan
    return x


def _repair(x: np.ndarray) -> np.ndarray:
    """Bridge non-finite samples by linear interpolation on the existing grid.

    No samples are inserted, removed or resampled, so the 256 Hz time base is kept.
    A channel with no finite samples becomes zeros.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return np.column_stack([_repair(x[:, j]) for j in range(x.shape[1])])
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {x.shape}")
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    repaired = x.copy()
    repaired[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return repaired


def _select_channels(samples: np.ndarray, channel_names) -> tuple[np.ndarray, np.ndarray]:
    """Return (ppg, imu) selected by WFDB channel name; imu may have zero columns."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D samples array, got shape {values.shape}")
    names = list(channel_names)
    if values.shape[1] != len(names):
        raise ValueError(f"Sample columns ({values.shape[1]}) != channel names ({len(names)})")
    column: dict[str, int] = {}
    for i, name in enumerate(names):
        column.setdefault(name, i)  # first occurrence wins
    if PPG_CHANNEL not in column:
        raise ValueError(f"Required channel '{PPG_CHANNEL}' not found in {names}")
    imu_cols = [column[n] for n in IMU_REFERENCE if n in column]
    return values[:, column[PPG_CHANNEL]], values[:, imu_cols]


# ============================================================ adaptive noise cancellation
def _tap_matrix(ref_padded: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Design matrix of delayed references: column c*TAPS + t holds channel c delayed by t."""
    n_ch = ref_padded.shape[1]
    X = np.empty((stop - start, TAPS * n_ch), dtype=np.float64)
    for c in range(n_ch):
        for t in range(TAPS):
            X[:, c * TAPS + t] = ref_padded[start + TAPS - t: stop + TAPS - t, c]
    return X


def _ridge_solve(gram: np.ndarray, target: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Ridge solution with a scale-relative strength, an absolute floor and a safe fallback."""
    lam = max(RIDGE * np.trace(gram) / gram.shape[0], RIDGE_ABS)
    system = gram + lam * identity
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, target, rcond=None)[0]


def _validated_block_prediction(X: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, float]:
    """Full-fit prediction plus an out-of-sample score (1 - residual / total variance).

    The block is split into alternating one-second chunks; each half is predicted from
    a fit on the other half.
    """
    identity = np.eye(X.shape[1])
    full = X @ _ridge_solve(X.T @ X, X.T @ desired, identity)
    chunk = max(1, int(CONFIDENCE_CHUNK_S * EXPECTED_FS))
    first = (np.arange(desired.size) // chunk) % 2 == 0
    second = ~first
    if first.sum() < X.shape[1] or second.sum() < X.shape[1]:
        return full, float("-inf")
    cross = np.empty_like(desired)
    cross[second] = X[second] @ _ridge_solve(X[first].T @ X[first], X[first].T @ desired[first], identity)
    cross[first] = X[first] @ _ridge_solve(X[second].T @ X[second], X[second].T @ desired[second], identity)
    centered = desired - np.mean(desired)
    total = float(centered @ centered)
    if total <= np.finfo(np.float64).eps:
        return full, float("-inf")
    residual = desired - cross
    return full, 1.0 - float(residual @ residual) / total


def _estimate_artifact(desired: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """IMU-predicted motion artifact, from blocks whose fit transfers out of sample.

    Blocks whose prediction fails the held-out check, or explains almost the whole PPG
    block, contribute nothing. Accepted blocks are blended with Hann weights.
    """
    n, n_ch = reference.shape
    if n == 0 or n_ch == 0:
        return np.zeros(n, dtype=np.float64)
    block = int(BLOCK_S * EXPECTED_FS)
    hop = block // 2
    window = np.hanning(block + 1)[:-1]
    ref_padded = np.vstack([np.tile(reference[0], (TAPS, 1)), reference])
    min_block = max(4 * TAPS * n_ch, 64)
    numerator = np.zeros(n, dtype=np.float64)
    denominator = np.zeros(n, dtype=np.float64)
    for start in range(0, n, hop):
        stop = min(start + block, n)
        if stop - start < min_block:
            continue
        d_block = desired[start:stop]
        prediction, confidence = _validated_block_prediction(_tap_matrix(ref_padded, start, stop), d_block)
        ratio = float(np.std(prediction)) / max(float(np.std(d_block)), np.finfo(np.float64).eps)
        if confidence <= CONFIDENCE_MIN or ratio >= ARTIFACT_RATIO_MAX:
            continue
        w = window[: stop - start]
        numerator[start:stop] += w * prediction
        denominator[start:stop] += w
    artifact = np.zeros(n, dtype=np.float64)
    valid = denominator >= 1e-9
    artifact[valid] = numerator[valid] / denominator[valid]
    return artifact


# ====================================================================== ridge features
def _ridge_features(desired: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """Dominant PPG frequency, its share of in-band power, and its maximum IMU coherence."""
    if desired.size < int(RIDGE_MIN_S * EXPECTED_FS) or not np.any(desired):
        return float("nan"), 0.0, 0.0
    nperseg = min(desired.size, RIDGE_NPERSEG)
    freqs, power = welch(desired, fs=EXPECTED_FS, nperseg=nperseg, noverlap=nperseg // 2)
    band = (freqs >= RIDGE_FMIN) & (freqs <= DES_BAND[1])
    in_band = np.flatnonzero(band)
    if in_band.size == 0:
        return float("nan"), 0.0, 0.0
    ridge_hz = float(freqs[in_band[np.argmax(power[in_band])]])
    total_power = float(np.sum(power[band]))
    if total_power <= np.finfo(np.float64).eps:
        return ridge_hz, 0.0, 0.0
    near_ridge = band & (np.abs(freqs - ridge_hz) <= RIDGE_HALF_WIDTH_HZ)
    power_share = float(np.sum(power[near_ridge])) / total_power
    max_coherence = 0.0
    for j in range(reference.shape[1]):
        if np.var(reference[:, j]) <= REFERENCE_VAR_EPS:
            continue
        coh_f, coh = coherence(desired, reference[:, j], fs=EXPECTED_FS,
                               nperseg=nperseg, noverlap=nperseg // 2)
        max_coherence = max(max_coherence, float(coh[np.argmin(np.abs(coh_f - ridge_hz))]))
    return ridge_hz, power_share, max_coherence


def _is_clean_low_rate_ridge(features: tuple[float, float, float]) -> bool:
    """A dominant low-frequency PPG ridge that the IMU does not share."""
    ridge_hz, power_share, max_coherence = features
    return bool(
        np.isfinite(ridge_hz)
        and ridge_hz < RIDGE_MAX_HZ
        and power_share >= RIDGE_POWER_MIN
        and max_coherence < RIDGE_COHERENCE_MAX
    )


# ===================================================================== pulse tracking
def _stft(x: np.ndarray):
    """STFT on the pulse-tracking grid (8 s window, 1 s hop)."""
    nper = int(STFT_WIN_S * EXPECTED_FS)
    hop = int(STFT_HOP_S * EXPECTED_FS)
    return stft(x, fs=EXPECTED_FS, nperseg=nper, noverlap=nper - hop, nfft=STFT_NFFT,
                boundary="even", padded=True)


def _normalised_power(Z: np.ndarray) -> np.ndarray:
    """Power per frame, scaled so each frame's maximum is 1."""
    P = np.abs(Z) ** 2
    return P / (P.max(axis=0, keepdims=True) + EPS)


def _viterbi_path(log_score: np.ndarray, df: float) -> np.ndarray:
    """Bin index per frame of the highest-scoring path with a smoothness penalty."""
    n_bins, n_frames = log_score.shape
    max_step = int(np.ceil(3 * PATH_STEP_HZ / df))
    offsets = np.arange(-max_step, max_step + 1)
    penalties = PATH_PENALTY * (offsets * df / PATH_STEP_HZ) ** 2
    bins = np.arange(n_bins)
    score = log_score[:, 0].copy()
    back = np.zeros((n_bins, n_frames), dtype=np.int64)
    for k in range(1, n_frames):
        best = np.full(n_bins, -np.inf)
        arg = np.zeros(n_bins, dtype=np.int64)
        for off, pen in zip(offsets, penalties):
            src = bins - off
            ok = (src >= 0) & (src < n_bins)
            cand = np.full(n_bins, -np.inf)
            cand[ok] = score[src[ok]] - pen
            better = cand > best
            best[better] = cand[better]
            arg[better] = src[better]
        score = best + log_score[:, k]
        back[:, k] = arg
    path = np.empty(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for k in range(n_frames - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    return path


def _track_pulse(signal: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-second pulse fundamental (Hz) and its confidence.

    PPG power is down-weighted where the IMU has power, then a Viterbi path picks the
    smoothest high-power route through PULSE_FMIN..PULSE_FMAX. Confidence is the share
    of in-band power lying within PATH_STEP_HZ of the path.
    """
    f, _, Z = _stft(signal)
    P = _normalised_power(Z)
    usable = [j for j in range(reference.shape[1]) if np.var(reference[:, j]) > REFERENCE_VAR_EPS]
    if usable:
        M = sum(_normalised_power(_stft(reference[:, j])[2]) for j in usable) / len(usable)
        M = M / (M.max(axis=0, keepdims=True) + EPS)
        P = P * np.clip(1.0 - MOTION_SUPPRESS * M, MOTION_FLOOR, 1.0)
    band = np.flatnonzero((f >= PULSE_FMIN) & (f <= PULSE_FMAX))
    fb, Pb = f[band], P[band]
    log_score = np.log(Pb / (Pb.max(axis=0, keepdims=True) + EPS) + 1e-3)
    path = _viterbi_path(log_score, fb[1] - fb[0])
    near_path = np.abs(fb[:, None] - fb[path][None, :]) <= PATH_STEP_HZ
    confidence = (Pb * near_path).sum(axis=0) / (Pb.sum(axis=0) + EPS)
    return fb[path], confidence


# =============================================================== two-peak correction
def _remove_double_peaks(y: np.ndarray, period_s: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Keep one pulse per pulse period.

    Of two peaks closer than DOUBLE_PEAK_FRAC x the local period (and never further
    apart than DOUBLE_PEAK_CAP_S), the smaller is flattened by a smooth cosine bridge
    between its neighbouring troughs. Where the pulse track is not confident both peaks
    are kept, since a wrong period would merge real beats.
    """
    y = y.copy()
    peaks, _ = find_peaks(y)
    troughs, _ = find_peaks(-y)
    if len(peaks) < 3:
        return y

    def trough_before(p: int) -> int:
        left = troughs[troughs < p]
        return int(left[-1]) if left.size else 0

    def trough_after(p: int) -> int:
        right = troughs[troughs > p]
        return int(right[0]) if right.size else len(y) - 1

    kept = [peaks[0]]
    for p in peaks[1:]:
        last = kept[-1]
        limit = min(DOUBLE_PEAK_FRAC * period_s[p], DOUBLE_PEAK_CAP_S) * EXPECTED_FS
        if p - last >= limit or confidence[p] < TRACK_CONFIDENCE_MIN:
            kept.append(p)
            continue
        drop = p if y[p] <= y[last] else last
        left, right = trough_before(drop), trough_after(drop)
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(right - left + 1) / max(right - left, 1))
        y[left:right + 1] = y[left] * (1 - w) + y[right] * w
        if drop == last:
            kept[-1] = p
    return y


# ============================================================================ pipeline
def clean_ppg(samples: np.ndarray, channel_names, sampling_rate: float = EXPECTED_FS) -> np.ndarray:
    """Return motion-suppressed PPG as a finite 1-D array of the input length."""
    if not np.isclose(sampling_rate, EXPECTED_FS, rtol=0.0, atol=1e-6):
        raise ValueError(f"Cleaner requires {EXPECTED_FS} Hz input; got {sampling_rate}")
    n = int(np.asarray(samples).shape[0])
    ppg, imu = _select_channels(samples, channel_names)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    desired = _bandpass(_repair(_mark_saturation(ppg)), *DES_BAND)
    has_imu = imu.shape[1] > 0
    reference = _bandpass(_repair(imu), *DES_BAND) if has_imu else imu
    features = _ridge_features(desired, reference)

    # Record-level gate: cancel only if the IMU moves and is coherent with the PPG;
    # otherwise the output is built from the cardiac-band PPG alone.
    cleaned = desired
    if (has_imu and np.max(np.var(reference, axis=0)) > REFERENCE_VAR_EPS
            and features[2] >= MOTION_COHERENCE_MIN):
        cleaned = desired - _estimate_artifact(desired, reference)

    final_band = LOW_RATE_BAND if _is_clean_low_rate_ridge(features) else FINAL_BAND
    output = _bandpass(cleaned, *final_band)

    # The pulse track needs at least two STFT windows of signal.
    if n >= int(2 * STFT_WIN_S * EXPECTED_FS):
        pulse_hz, confidence = _track_pulse(_bandpass(cleaned, *DES_BAND), reference)
        t = np.arange(n) / EXPECTED_FS
        frames = np.arange(len(pulse_hz)) * STFT_HOP_S
        period_s = np.interp(t, frames, 1.0 / np.clip(pulse_hz, *PULSE_CLIP_HZ))
        output = _remove_double_peaks(output, period_s, np.interp(t, frames, confidence))
    return _finalize(output, n)


def _finalize(output: np.ndarray, n: int) -> np.ndarray:
    """Enforce the output contract: 1-D, exactly n samples, all finite."""
    out = np.asarray(output, dtype=np.float64).reshape(-1)[:n]
    if out.size < n:
        out = np.pad(out, (0, n - out.size), constant_values=out[-1] if out.size else 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ================================================================================ CLI
def _write_output(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        np.savetxt(path, values, delimiter=",")
    else:
        np.save(path, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_record", type=Path, help="WFDB record base path or .hea/.dat path")
    parser.add_argument("output", type=Path, help="output .npy or single-column .csv path")
    args = parser.parse_args()

    record_path = args.input_record
    if record_path.suffix.lower() in (".hea", ".dat"):
        record_path = record_path.with_suffix("")
    record = wfdb.rdrecord(str(record_path), physical=True)
    cleaned = clean_ppg(record.p_signal, record.sig_name, float(record.fs))
    _write_output(args.output, cleaned)
    print(f"Wrote {cleaned.size} finite samples to {args.output}")


if __name__ == "__main__":
    main()
