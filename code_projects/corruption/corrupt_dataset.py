#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corrupt_subset.py

Apply time-domain, frequency-domain, and spectrogram-domain corruptions
on the TUH-SZ test set, after preprocessing/segmentation from tusz_utilities.
Run with:
    python corrupt_dataset.py
"""
import os
import numpy as np
from scipy.signal import ShortTimeFFT, get_window
from scipy.fft import rfft, irfft
from scipy.io import savemat
from code_projects.data.tusz_utilities import get_montages_from_edf, preprocess_data, calculate_spectrogram

# === User Configuration ===
DATA_DIR      = '/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Raw_data/eval'
OUTPUT_DIR    = '/work/groups/kismed/Datasets/TUH_Seizure_Corpus/TUSZ-C'
RESAMPLE_FREQ = 250.0                # Hz
SEGMENT_LEN   = 250 * 10             # samples (10s @250Hz)
NOTCH_FREQ    = 50.0                 # Hz
NOTCH_Q       = 30.0                 # quality factor
NPERSEG       = 256
NOVERLAP      = 128
TIME_CORRS    = ['gaussian_noise', 'signal_dropout', 'channel_dropout', 'flip']
FREQ_CORRS    = ['gaussian_noise']
TF_CORRS      = ['spectral_aug']

# ----------------------------------------------------------------------------
# Severity parameter generator (five levels)
# ----------------------------------------------------------------------------
def severity_params(corruption: str, severity: int, fs: float) -> dict:
    if corruption == 'gaussian_noise':
        return {'std': 0.1 * severity}
    if corruption == 'signal_dropout':
        # relative dropout up to 10-20% of the segment at max severity
        L        = SEGMENT_LEN               # 250*10 = 2500 samples
        frac_min = 0.1 * (severity / 5.0)    
        frac_max = 0.2 * (severity / 5.0)     
        return {
            'dropout_prob': 0.05 * severity,
            'min_length':   int(frac_min * L),
            'max_length':   int(frac_max * L)
        }
    if corruption == 'channel_dropout':
        return {'dropout_prob': min(1.0, 0.2 * severity)}
    if corruption == 'flip':
        return {}
    if corruption == 'spectral_aug':
        return {
            'band':      'alpha',            # choose 'delta','theta','beta','gamma'
            'reduction': 1.0 - 0.1 * severity,
            'prob':      0.1 * severity
        }
    raise ValueError(f"Unknown corruption: {corruption}")

# ----------------------------------------------------------------------------
# Corruption functions
# ----------------------------------------------------------------------------
def add_gaussian_noise_td(x: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std * np.std(x), size=x.shape)
    return x + noise

def apply_signal_dropout(x: np.ndarray, dropout_prob: float,
                         min_length: int, max_length: int) -> np.ndarray:
    y = x.copy()
    L = len(x)
    n = int((dropout_prob * L) / max(min_length,1)) or int(dropout_prob > 0)
    starts = np.random.choice(max(L-max_length,1), size=n, replace=False)
    lengths = np.random.randint(min_length, max_length+1, size=n)
    for s, le in zip(starts, lengths): y[s:s+le] = 0
    return y

def apply_channel_dropout(x: np.ndarray, dropout_prob: float) -> np.ndarray:
    return np.zeros_like(x) if np.random.rand() < dropout_prob else x

def apply_flip(x: np.ndarray) -> np.ndarray:
    return -x

def add_gaussian_noise_fd(x: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std * np.std(x), size=x.shape)
    X = rfft(x)
    N = rfft(noise)
    return irfft(X + N, n=x.shape[0])

def apply_spectral_augmentation_tf(S: np.ndarray,
                                  f: np.ndarray, t: np.ndarray,
                                  band: str, reduction: float, prob: float) -> np.ndarray:
    ranges = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,12),'beta':(12,30),'gamma':(30,80)}
    if band not in ranges: raise ValueError(f"Unsupported band '{band}'")
    low, high = ranges[band]
    mask_f = (f>=low)&(f<high)
    mask_t = np.random.rand(len(t)) < prob
    S2 = S.copy()
    S2[mask_f[:,None] & mask_t[None,:]] *= reduction
    return S2

# ----------------------------------------------------------------------------
# Main corruption pipeline
# ----------------------------------------------------------------------------
def corrupt_and_save():
    win = get_window('hann', NPERSEG)
    hop = NPERSEG - NOVERLAP
    STFT = ShortTimeFFT(win, hop=hop, fs=RESAMPLE_FREQ)
    notch_args = {'w0': NOTCH_FREQ, 'Q':  NOTCH_Q}

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith('.edf'): 
                continue
            edf_path = os.path.join(root, fname)
            montages, fs, ch_names = get_montages_from_edf(edf_path)
            if montages is None:
                continue

            segments = preprocess_data(
                montages, fs, notch_args,
                RESAMPLE_FREQ, SEGMENT_LEN
            )
            rel = os.path.relpath(root, DATA_DIR)
            base = os.path.splitext(fname)[0]

            for corruption in TIME_CORRS + FREQ_CORRS + TF_CORRS:
                for sev in range(1,6):
                    params = severity_params(corruption, sev, RESAMPLE_FREQ)
                    segs_c = segments.copy()

                    # time-domain
                    if corruption in TIME_CORRS:
                        fn = {
                            'gaussian_noise':  add_gaussian_noise_td,
                            'signal_dropout':  apply_signal_dropout,
                            'channel_dropout': apply_channel_dropout,
                            'flip':            apply_flip
                        }[corruption]
                        for i in range(segs_c.shape[0]):
                            for ch in range(segs_c.shape[1]):
                                segs_c[i,ch] = fn(segs_c[i,ch], **params)

                    # freq-domain
                    if corruption in FREQ_CORRS:
                        for i in range(segs_c.shape[0]):
                            for ch in range(segs_c.shape[1]):
                                segs_c[i,ch] = add_gaussian_noise_fd(segs_c[i,ch], **params)

                    # compute spectrograms
                    specs = calculate_spectrogram(segs_c, STFT, max_n_parallel=500)
                    specs = specs.detach().cpu().numpy()
                    specs = specs.reshape(
                        segs_c.shape[0],
                        len(ch_names),
                        specs.shape[2],
                        specs.shape[3]
                    )

                    # time-frequency
                    if corruption in TF_CORRS:
                        f_vals = np.linspace(0, RESAMPLE_FREQ/2, specs.shape[2])
                        t_vals = np.linspace(0, SEGMENT_LEN/RESAMPLE_FREQ, specs.shape[3])
                        for i in range(specs.shape[0]):
                            for ch in range(specs.shape[1]):
                                specs[i,ch] = apply_spectral_augmentation_tf(
                                    specs[i,ch], f_vals, t_vals, **params
                                )

                    # now save each segment channel with its own .mat filename
                    out_sub = os.path.join(
                        OUTPUT_DIR,
                        f"severity_{sev}",
                        corruption,
                        rel
                    )
                    os.makedirs(out_sub, exist_ok=True)

                    for i in range(segs_c.shape[0]):
                        for ch in range(segs_c.shape[1]):
                            raw_seg  = segments[i, ch]
                            corr_seg = segs_c[i, ch]
                            spec     = specs[i, ch]

                            # build a unique filename
                            fname_mat = f"{base}_seg{i}_ch{ch}.mat"
                            out_path  = os.path.join(out_sub, fname_mat)

                            savemat(out_path, {
                                'segment_raw':  raw_seg,
                                'segment_corr': corr_seg,
                                'spectrogram':  spec,
                                'channels':     ch_names,
                                'fs':           RESAMPLE_FREQ
                            })

    print("All corruptions applied and saved.")

if __name__ == '__main__':
    corrupt_and_save()