#!/usr/bin/env python3
# visualize_preprocess.py

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import resample, stft, get_window

from code_projects.data.tusz_utilities import (
    get_montages_from_edf, 
    apply_notch_filter,
    split_into_segments   
)

# ── 1) CONFIG ──────────────────────────────────────────────────────────────
EDF_FILE = (
    "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Raw_data/train/aaaaadyf/s001_2006_03_23/02_tcp_le/"
    "aaaaadyf_s001_t001.edf"
)
# pick the first montage channel (e.g. "FP1-F7")
MONTAGE_IDX = 0  

# pipeline params
NOTCH_ARGS    = {"w0": 60.0, "Q": 30.0}
RESAMPLE_FS   = 200.0    # Hz
SEGMENT_LEN   = 1024     # samples
WINDOW_SIZE   = 256
NOVERLAP      = 12
HOP           = WINDOW_SIZE - NOVERLAP

# ── 2) LOAD + MONTAGE ───────────────────────────────────────────────────────
# read EDF
raw = mne.io.read_raw_edf(EDF_FILE, preload=True, verbose=False)
# compute bipolar montages (shape: [n_montages, n_samples])
montages, fs, montage_names = get_montages_from_edf(EDF_FILE)

# pick one montage channel and convert to μV
signal_uv = montages[MONTAGE_IDX] * 1e6  # from V to μV
channel_name = montage_names[MONTAGE_IDX]

# ── 3) NOTCH FILTER ─────────────────────────────────────────────────────────
filtered_uv = apply_notch_filter(signal_uv[None, :], fs, NOTCH_ARGS)[0]

# ── 4) RESAMPLE ─────────────────────────────────────────────────────────────
n_out = int(len(filtered_uv) * RESAMPLE_FS / fs)
resamp_uv = resample(filtered_uv, n_out)

# ── 5) SPLIT INTO SEGMENTS ─────────────────────────────────────────────────
segments = split_into_segments(resamp_uv[None, :], SEGMENT_LEN)
seg = segments[0]  # first segment

# ── 6) COMPUTE SPECTROGRAM ──────────────────────────────────────────────────
f, t, Z = stft(seg, fs=RESAMPLE_FS, window="hann",
               nperseg=WINDOW_SIZE, noverlap=NOVERLAP)
Z = Z.squeeze(0)
spec_db = 20 * np.log10(np.abs(Z) + 1e-6)  # in dB

# ── 7) PLOT + SAVE SVG ──────────────────────────────────────────────────────
fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=False)

# time axes
times_raw   = np.arange(signal_uv.shape[0]) / fs
times_resamp= np.arange(resamp_uv.shape[0]) / RESAMPLE_FS

axs[0].plot(times_raw, signal_uv, lw=0.8)
axs[0].set_title(f"Raw Bipolar Montage ({channel_name}) in μV")
axs[0].set_ylabel("μV")

axs[1].plot(times_raw, filtered_uv, lw=0.8)
axs[1].set_title("Notch-Filtered (60 Hz)")
axs[1].set_ylabel("μV")

axs[2].plot(times_resamp, resamp_uv, lw=0.8)
axs[2].set_title(f"Resampled at {RESAMPLE_FS:.0f} Hz")
axs[2].set_ylabel("μV")

im = axs[3].pcolormesh(
    t, f, spec_db,
    shading="gouraud"
)
axs[3].set_title("Spectrogram (dB)")
axs[3].set_ylabel("Frequency (Hz)")
axs[3].set_xlabel("Time (s)")
fig.colorbar(im, ax=axs[3], label="dB")

plt.tight_layout()
plt.savefig("preprocessing_pipeline.svg", format="svg")
plt.show()
