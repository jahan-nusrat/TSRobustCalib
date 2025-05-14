#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each corruption type, load one preprocessed segment (raw & corrupted
at severities 2 & 5) and overlay it on the full raw EDF trace, showing:

  • Full recording with overlaid segment (raw + corrupted)
  • Zoom-in on that 10 s window (raw + corrupted)

Each figure is saved as an SVG in OUTPUT_DIR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ─── User configuration ────────────────────────────────────────────────────────
RAW_DATA_ROOT  = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Raw_data/eval"
CORRUPTED_ROOT = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/TUSZ-C"
REL_PATH       = "aaaaasip/s001_2015_01_29/01_tcp_ar/aaaaasip_s001_t000_seg101_ch7.mat"

# Which severities to compare
SEVERITIES     = [2, 5]

# List of corruption folder names
CORRUPTIONS    = [
    "gaussian_noise",
    "signal_dropout",
    "channel_dropout",
    "flip",
    "spectral_aug"
]

# Preprocessing parameters (must match corrupt_subset.py)
FS  = 250.0      # Hz
SEGMENT_LEN    = 250 * 10 

# Output directory for figures
OUTPUT_DIR     = "/work/home/nj31voho/thesis/eeg/models/Corrupt/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # build relative time axis: 0 → SEGMENT_LEN/FS seconds
    t_rel = np.arange(SEGMENT_LEN) / FS

    for corr in CORRUPTIONS:
        # load raw segment (stored under severity_1)
        raw_path = os.path.join(CORRUPTED_ROOT, "severity_1", corr, REL_PATH)
        raw_mat  = loadmat(raw_path)
        seg_raw  = raw_mat["segment_raw"].squeeze()

        # load corrupted segments at each severity
        corr_segs = {}
        for sev in SEVERITIES:
            mat = loadmat(os.path.join(
                CORRUPTED_ROOT,
                f"severity_{sev}",
                corr,
                REL_PATH
            ))
            corr_segs[sev] = mat["segment_corr"].squeeze()

        # create 1×3 figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
        fig.suptitle(f"{corr.replace('_',' ').title()} – 10 s EEG Segment", fontsize=16)

        # plot settings
        titles = ["Raw Segment"] + [f"Severity {sev}" for sev in SEVERITIES]
        data   = [seg_raw] + [corr_segs[sev] for sev in SEVERITIES]
        colors = ["C0", "C1", "C2"]

        for ax, sig, title, c in zip(axes, data, titles, colors):
            ax.plot(t_rel, sig, color=c, linewidth=0.8)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_xlim(0, SEGMENT_LEN/FS)
            ax.grid(alpha=0.3)

        # save figure
        svg_name = f"{corr}_10s_segment.svg"
        out_path = os.path.join(OUTPUT_DIR, svg_name)
        fig.savefig(out_path, format="svg")
        print(f"Saved {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()