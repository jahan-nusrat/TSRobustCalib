import numpy as np
import mne
from scipy.io import savemat
from scipy.signal import ShortTimeFFT, iirnotch, filtfilt, resample, get_window
from utils.constants import *
import torch
from typing import List, Any, Tuple
import os
import pandas as pd

def getOrderedChannels(file_name: str, verbose: bool, labels: list[str], channel_names: list[str]) -> list[int]:
    """
    Map expected channel names to their indices in the actual EDF file's channel list.

    Args:
        file_name: Path to the EDF file (used for verbose logging)
        verbose: Whether to print detailed output
        labels: Actual channel labels from the EDF file (e.g. from f.ch_names)
        channel_names: Ordered list of expected channel names (e.g. ORDERED_CHANNELS)

    Returns:
        List of indices in `labels` matching the expected channels.
    """
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0].strip()

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except ValueError:
            if verbose:
                print(f"{file_name}: Missing required channel '{ch}'")
            raise Exception("channel not match")
    return ordered_channels


def getEDFsignals(edf) -> np.ndarray:
    """
    Get EEG signal from MNE Raw object.

    Args:
        edf: MNE Raw object

    Returns:
        signals: shape (num_channels, num_data_points)
    """
    return edf.get_data()


def split_into_segments(data:np.ndarray, segment_len:int) -> np.ndarray:
    target_size = int(segment_len * np.ceil(data.shape[-1] / segment_len))
    data = np.pad(data, ((0, 0), (int(np.floor((target_size-data.shape[-1])/2)), int(np.ceil((target_size-data.shape[-1])/2))))) # Pad zeros right and left
    data = np.split(data, indices_or_sections=data.shape[1]//segment_len, axis=1) 
    data = np.stack(data, axis=0) # Shape : N_segments x N_channels x Segment_len
    return data

def apply_notch_filter(data:np.ndarray, fs:float, notch_args:dict[str, Any]):
    b, a = iirnotch(fs=fs, **notch_args)
    return filtfilt(b, a, data, axis=-1)

def preprocess_data(data:np.ndarray, fs:float, notch_args:dict[str, Any], resample_freq:float, segment_len:int) -> np.ndarray:
    data = apply_notch_filter(data, fs, notch_args) # Shape : N_channels x N_datapoints
    data = resample(data, num=int(data.shape[1]*resample_freq/fs), axis=-1) # Shape : N_channels x N_datapoints_new
    data = split_into_segments(data, segment_len) # Shape : N_segments x N_channels x Segment_len 
    data = (data-data.mean(axis=2, keepdims=True)) / (data.std(axis=2, keepdims=True) + 1e-30) # Standardize and avoid division through zero
    return data

def calculate_spectrogram(data:np.ndarray, STFT:ShortTimeFFT, max_n_parallel:int) -> torch.Tensor:
    spectograms = []
    for i in range(0, data.shape[0], max_n_parallel):                                 # Shape : N_segments x N_channels x N_bins x N_timepoints
        spectograms.append(np.abs(STFT.stft(data[i:i+max_n_parallel, ...], axis=-1))) # Need to split the data up into max 500 segments each for STFT otherwise it takes forever
    spectograms = np.concatenate(spectograms, axis=0)
    spectograms = spectograms.reshape(-1, spectograms.shape[2], spectograms.shape[3]) # Shape : N_segments * N_channels x N_bins x N_timepoints

    # Min-Maxing
    spectograms -= spectograms.min(axis=(1, 2), keepdims=True)
    spectograms /= (spectograms.max(axis=(1, 2), keepdims=True) + 1e-30) # Range : [0, 1] and avoid division through zero

    return torch.from_numpy(spectograms).unsqueeze(1).float()

def generate_label_TUSZ(label_path: str, N_segments: int, fs: float, segment_len: int, montage_names: list[str]) -> torch.Tensor:
    df = pd.read_csv(label_path, comment='#')
    labels = torch.zeros((N_segments, len(montage_names)), dtype=int)

    segment_duration = segment_len / fs
    segment_times = [(i * segment_duration, (i + 1) * segment_duration) for i in range(N_segments)]

    for i in range(len(df)):
        channel, onset, offset, label = df[["channel", "start_time", "stop_time", "label"]].iloc[i]
        label_int = CONST_TUSZ_CLASS_NAME_TO_LABEL_SEIZURE_NORMAL[label.upper()]
        if label_int == Label_seizure_normal.NORMAL:
            continue

        # Ensure the channel is in montage_names
        channel_up = channel.upper()
        if channel_up not in montage_names:
            print(f"Skipping unknown montage label: {channel_up}")
            continue
        
        idx_channel = montage_names.index(channel_up)
        seizure_start = onset
        seizure_end = offset
        
        for seg_idx, (seg_start, seg_end) in enumerate(segment_times):
            overlap = max(0, min(seg_end, seizure_end) - max(seg_start, seizure_start))
            if overlap / segment_duration >= 0.5:  # or your chosen threshold
                labels[seg_idx, idx_channel] = label_int.value

    return labels.float()


def get_montages_from_edf(edf_path: str) -> Tuple[np.ndarray, float, List[str]]:
    try:
        f = mne.io.read_raw_edf(edf_path, preload=True)
    except Exception as e:
        print(f"Failed to load EDF: {edf_path}: {e}")
        return None, None, None

    # Always use the reduced montage (19 channels) for consistency
    print(f"Using REDUCED montage for {edf_path}")
    channels = ORDERED_CHANNELS_REDUCED
    montage_indices = MONTAGES_CHANNELS_IDX_REDUCED
    active_montage_names = MONTAGES_NAMES_REDUCED

    try:
        ordered_channels = getOrderedChannels(edf_path, False, f.ch_names, channels)
    except Exception as e:
        print(f"Skipping {edf_path} due to channel mismatch: {e}")
        return None, None, None

    signals = f.get_data()[ordered_channels, :]
    fs = f.info['sfreq']
    f.close()

    computed_montages = []
    computed_names = []
    for name, (i, j) in zip(active_montage_names, montage_indices):
        try:
            computed_montages.append(signals[i, :] - signals[j, :])
            computed_names.append(name)
        except IndexError:
            print(f"Montage {name} not available, skipping.")
    
    if not computed_montages:
        return None, fs, None

    montage_array = np.array(computed_montages)
    return montage_array, fs, computed_names



def tusz_to_segment_mat(
    data_path: str,
    save_path: str,
    resample_freq: float,
    segment_len: int,
    spectrogram_args: dict,
    notch_args: dict
) -> None:
    import os
    import torch
    from scipy.signal import ShortTimeFFT, get_window
    from scipy.io import savemat

    # Prepare STFT parameters
    window_size = spectrogram_args["nperseg"]
    noverlap = spectrogram_args["noverlap"]
    hop_size = window_size - noverlap
    STFT = ShortTimeFFT(get_window('hann', window_size), hop=hop_size, fs=resample_freq)

    seizure_count = 0
    normal_count = 0
    total_files_checked = 0

    # Process each EDF file
    for root, subdirs, files in os.walk(data_path):
        for name in files:
            if not name.lower().endswith(".edf"):
                continue

            edf_path = os.path.join(root, name)
            save_path_0 = os.path.join(save_path, name.lower().replace(".edf", "_0.mat"))
            if os.path.exists(save_path_0):
                print(f"Skipped {edf_path} as it has already been processed")
                continue

            # Load EDF and extract montages. Updated get_montages_from_edf returns (montage_array, fs, montage_names)
            montages, fs, montage_names = get_montages_from_edf(edf_path)
            if montages is None:
                continue

            # Preprocess: apply notch filtering, resample, and split into segments.
            montages = preprocess_data(montages, fs, notch_args, resample_freq, segment_len)

            # Generate labels
            csv_path = edf_path.replace(".edf", ".csv")
            labels = generate_label_TUSZ(csv_path, montages.shape[0], fs, segment_len, montage_names)

            total_files_checked += 1
            if torch.any(labels > 0):
                seizure_count += 1
                print(f"[{total_files_checked}] {edf_path} contains seizure")
            else:
                normal_count += 1
                print(f"[{total_files_checked}] {edf_path} normal EEG")

            # Calculate spectrograms from the preprocessed montages
            spectrograms = calculate_spectrogram(montages, STFT, max_n_parallel=500)
            spectrograms = spectrograms.reshape(
                montages.shape[0], len(montage_names), spectrograms.shape[2], spectrograms.shape[3]
            )

            # Save each segment as a .mat file.
            for i in range(spectrograms.shape[0]):
                rel_path = os.path.relpath(edf_path, data_path)
                rel_dir = os.path.dirname(rel_path)
                subfolder_save_path = os.path.join(save_path, rel_dir)
                os.makedirs(subfolder_save_path, exist_ok=True)
                base_name = name.lower().replace(".edf", f"_{i}.mat")
                cur_save_path = os.path.join(subfolder_save_path, base_name)
                data_dict = {
                    "spectrogram": spectrograms[i, ...],
                    "label": labels[i, ...],
                    "channels": montage_names,
                    "fs": resample_freq
                }
                savemat(cur_save_path, data_dict)

    if total_files_checked > 0:
        seizure_percent = (seizure_count / total_files_checked) * 100
        normal_percent = (normal_count / total_files_checked) * 100
        print("\n=== Dataset Summary ===")
        print(f"Total EDF files processed: {total_files_checked}")
        print(f"  Seizure files: {seizure_count} ({seizure_percent:.2f}%)")
        print(f"  Normal files:  {normal_count} ({normal_percent:.2f}%)")
    else:
        print("No EDF files were processed.")
