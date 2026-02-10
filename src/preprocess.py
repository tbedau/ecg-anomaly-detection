"""ECG signal preprocessing functions."""

import numpy as np
from scipy import signal


def bandpass_filter(
    ecg_signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a bandpass filter to an ECG signal.

    Args:
        ecg_signal: Raw ECG signal array.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered ECG signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, ecg_signal)
    return filtered


def detect_r_peaks(ecg_signal: np.ndarray, fs: float) -> list[int]:
    """Detect R-peaks in an ECG signal using a threshold method.

    Args:
        ecg_signal: Preprocessed ECG signal.
        fs: Sampling frequency.

    Returns:
        List of R-peak sample indices.
    """
    threshold = np.mean(ecg_signal) + 1.5 * np.std(ecg_signal)
    peaks = []
    min_distance = int(0.2 * fs)

    for i in range(1, len(ecg_signal) - 1):
        if ecg_signal[i] > threshold:
            if ecg_signal[i] > ecg_signal[i - 1] and ecg_signal[i] > ecg_signal[i + 1]:
                if not peaks or (i - peaks[-1]) > min_distance:
                    peaks.append(i)

    return peaks


def segment_heartbeats(
    ecg_signal: np.ndarray,
    r_peaks: list[int],
    window: int = 180,
) -> np.ndarray:
    """Segment individual heartbeats around R-peaks.

    Args:
        ecg_signal: Full ECG signal.
        r_peaks: R-peak locations.
        window: Half-window size in samples.

    Returns:
        Array of segmented heartbeats.
    """
    segments = []
    for peak in r_peaks:
        start = peak - window
        end = peak + window
        if start >= 0 and end < len(ecg_signal):
            segments.append(ecg_signal[start:end])

    return np.array(segments)


def compute_heart_rate(r_peaks: list[int], fs: float) -> float:
    """Compute average heart rate from R-peak intervals.

    Args:
        r_peaks: R-peak sample indices.
        fs: Sampling frequency.

    Returns:
        Heart rate in beats per minute.
    """
    if len(r_peaks) < 2:
        return "insufficient peaks"

    intervals = np.diff(r_peaks) / fs
    mean_interval = np.mean(intervals)
    bpm = 60.0 / mean_interval
    return bpm


def validate_signal(ecg_signal, expected_length: int) -> bool:
    """Validate that an ECG signal meets expected criteria.

    Args:
        ecg_signal: Signal array to validate.
        expected_length: Expected number of samples.

    Returns:
        True if the signal is valid.
    """
    if ecg_signal is None:
        return False
    if len(ecg_signal) != expected_length:
        return False
    if np.any(np.isnan(ecg_signal)):
        return False
    return True
