import numpy as np

def clean_vitals(hr_series: np.ndarray, spo2_series: np.ndarray):
    """
    Rejects physiological outliers and clips signals to realistic human bounds.
    HR: 35 - 220 bpm | SpO2: 70 - 100%
    """
    clean_hr = np.clip(hr_series, 35.0, 220.0)
    clean_spo2 = np.clip(spo2_series, 70.0, 100.0)
    return clean_hr, clean_spo2

def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Applies a moving average filter to smooth raw sensor noise."""
    if len(signal) < window_size:
        return signal
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def normalize_vitals(hr: np.ndarray, spo2: np.ndarray):
    """
    Normalizes physiological parameters to [0, 1] range:
    HR mapped across [40, 200] bpm, SpO2 mapped across [70, 100]%.
    """
    norm_hr = (hr - 40.0) / (200.0 - 40.0)
    norm_spo2 = (spo2 - 70.0) / (100.0 - 70.0)
    return np.clip(norm_hr, 0.0, 1.0), np.clip(norm_spo2, 0.0, 1.0)