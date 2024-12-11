import os
import numpy as np
from scipy.signal import spectrogram, iirnotch, filtfilt, resample
from pyedflib import EdfReader
from constants import ORDERED_CHANNELS, MONTAGES_CHANNELS_IDX, SEIZURE_LABEL_MAPPING

def load_edf_signals(file_path):
    """Load signals from EDF and reorder channels."""
    edf = EdfReader(file_path)
    labels = edf.getSignalLabels()
    print(f"EDF Labels: {labels}")  # Debugging: Print the labels in the EDF file

    # Find indices of the channels in ORDERED_CHANNELS
    ordered_indices = [labels.index(ch) for ch in ORDERED_CHANNELS if ch in labels]
    if not ordered_indices:
        raise ValueError(f"No matching channels found in {file_path}. Available labels: {labels}")

    # Load signals for the ordered indices
    signals = np.array([edf.readSignal(i) for i in ordered_indices])

    # Create montages
    montages = np.stack([signals[i] - signals[j] for i, j in MONTAGES_CHANNELS_IDX], axis=0)
    fs = edf.getSampleFrequency(ordered_indices[0])  # Assume all channels have the same sampling rate
    edf.close()
    return montages, fs

    """Load signals from EDF and reorder channels."""
    edf = EdfReader(file_path)
    fs = edf.getSampleFrequency(0)  # Sampling rate
    labels = edf.getSignalLabels()
    
    # Get ordered signals
    ordered_indices = [labels.index(ch) for ch in ORDERED_CHANNELS if ch in labels]
    signals = np.array([edf.readSignal(i) for i in ordered_indices])

    # Create montages
    montages = np.stack([signals[i] - signals[j] for i, j in MONTAGES_CHANNELS_IDX], axis=0)
    edf.close()
    return montages, fs

def apply_notch_filter(data, fs, freq=50.0, Q=30.0):
    """Apply notch filter to remove powerline noise."""
    b, a = iirnotch(w0=freq / (fs / 2), Q=Q)
    return filtfilt(b, a, data, axis=-1)

def preprocess_signals(data, fs, target_fs=100, segment_length=2):
    """Resample, segment, normalize EEG data, and generate spectrograms."""
    # Resample to the target frequency
    resampled_data = resample(data, num=int(data.shape[1] * target_fs / fs), axis=-1)

    # Segment the data
    segment_samples = target_fs * segment_length
    n_segments = resampled_data.shape[1] // segment_samples
    segments = resampled_data[:, :n_segments * segment_samples].reshape(
        n_segments, data.shape[0], segment_samples
    )

    # Generate spectrograms for each segment
    spectrograms = []
    for segment in segments:
        _, _, Sxx = spectrogram(segment, fs=target_fs, nperseg=128)
        spectrograms.append(Sxx)
    spectrograms = np.array(spectrograms)  # Shape: (n_segments, n_channels, freq_bins, time_bins)

    # Normalize spectrograms and ensure correct dimensions
    spectrograms = np.array(spectrograms)  # Shape: (n_segments, n_channels, freq_bins, time_bins)
    spectrograms = np.expand_dims(spectrograms, axis=1)  # Add channel dimension -> (n_segments, 1, freq_bins, time_bins)
    spectrograms = (spectrograms - spectrograms.mean(axis=(2, 3), keepdims=True)) / (
        spectrograms.std(axis=(2, 3), keepdims=True) + 1e-6
    )

    return spectrograms

def generate_labels(csv_path, n_segments, segment_length, fs):
    """
    Generate labels for each EEG segment based on annotations in the .csv file.
    :param csv_path: Path to the annotation file.
    :param n_segments: Number of segments in the EEG data.
    :param segment_length: Duration of each segment in seconds.
    :param fs: Sampling rate in Hz.
    :return: A NumPy array of labels for each segment.
    """
    labels = np.zeros(n_segments, dtype=int)  # Default to NORMAL

    try:
        # Read the CSV file and skip comment lines
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # Filter out lines starting with '#' or containing the header
        annotations = [line.strip().split(',') for line in lines if not line.startswith('#') and 'start_time' not in line]

        for row in annotations:
            try:
                # Extract start_time, stop_time, and label
                start_time = float(row[1])  # Second column
                stop_time = float(row[2])  # Third column
                label = row[3].strip().upper()  # Fourth column

                # Map the label using SEIZURE_LABEL_MAPPING
                if label in SEIZURE_LABEL_MAPPING:
                    label_value = SEIZURE_LABEL_MAPPING[label].value

                    # Convert times to segment indices
                    start_idx = int(start_time // segment_length)
                    stop_idx = int(stop_time // segment_length)

                    # Assign the label to the corresponding segments
                    labels[start_idx:stop_idx + 1] = label_value
            except ValueError as e:
                print(f"Error parsing row {row}: {e}")
                continue
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

    return labels

    """
    Generate labels for each EEG segment based on annotations in the .csv file.
    :param csv_path: Path to the annotation file.
    :param n_segments: Number of segments in the EEG data.
    :param segment_length: Duration of each segment in seconds.
    :param fs: Sampling rate in Hz.
    :return: A NumPy array of labels for each segment.
    """
    labels = np.zeros(n_segments, dtype=int)  # Default to NORMAL

    try:
        # Read the CSV file and skip comments starting with '#'
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # Filter out lines starting with '#'
        annotations = [line.strip().split(',') for line in lines if not line.startswith('#')]

        for row in annotations:
            # Extract start_time, stop_time, and label
            try:
                start_time = float(row[1])  # Second column
                stop_time = float(row[2])  # Third column
                label = row[3].strip().upper()  # Fourth column

                # Map the label using SEIZURE_LABEL_MAPPING
                if label in SEIZURE_LABEL_MAPPING:
                    label_value = SEIZURE_LABEL_MAPPING[label].value

                    # Convert times to segment indices
                    start_idx = int(start_time // segment_length)
                    stop_idx = int(stop_time // segment_length)

                    # Assign the label to the corresponding segments
                    labels[start_idx:stop_idx + 1] = label_value
            except ValueError as e:
                print(f"Error parsing row {row}: {e}")
                continue
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

    return labels

    """
    Generate labels for each EEG segment based on annotations in the .csv file.
    :param csv_path: Path to the annotation file.
    :param n_segments: Number of segments in the EEG data.
    :param segment_length: Duration of each segment in seconds.
    :param fs: Sampling rate in Hz.
    :return: A NumPy array of labels for each segment.
    """
    labels = np.zeros(n_segments, dtype=int)  # Default to NORMAL

    try:
        # Load the annotation file, skipping the comment lines
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # Filter out comment lines starting with '#'
        annotations = [line.strip().split(',') for line in lines if not line.startswith('#')]

        for row in annotations:
            # Extract start_time, stop_time, and label
            start_time = float(row[1])  # Second column
            stop_time = float(row[2])  # Third column
            label = row[3].strip().upper()  # Fourth column

            # Map the label using SEIZURE_LABEL_MAPPING
            if label in SEIZURE_LABEL_MAPPING:
                label_value = SEIZURE_LABEL_MAPPING[label].value

                # Convert times to segment indices
                start_idx = int(start_time // segment_length)
                stop_idx = int(stop_time // segment_length)

                # Assign the label to the corresponding segments
                labels[start_idx:stop_idx + 1] = label_value
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

    return labels

    """
    Generate labels for each EEG segment based on annotations in the .csv file.
    """
    labels = np.zeros(n_segments, dtype=int)  # Default to NORMAL
    try:
        # Load the annotation file, skipping the header row
        annotations = np.genfromtxt(csv_path, delimiter=',', dtype=None, encoding='utf-8', skip_header=1)
        
        # Ensure annotations are processed correctly
        for row in annotations:
            start_time, stop_time, label = float(row[1]), float(row[2]), row[3]
            if label.upper() in SEIZURE_LABEL_MAPPING:
                label_value = SEIZURE_LABEL_MAPPING[label.upper()].value
                start_idx = int(start_time * fs // segment_length)
                stop_idx = int(stop_time * fs // segment_length)
                labels[start_idx:stop_idx + 1] = label_value
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
    return labels

    """Generate labels for each EEG segment."""
    labels = np.zeros(n_segments, dtype=int)
    annotations = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=str)
    for row in annotations:
        start_time, stop_time, label = float(row[1]), float(row[2]), row[3]
        if label.upper() in SEIZURE_LABEL_MAPPING:
            label_value = SEIZURE_LABEL_MAPPING[label.upper()].value
            start_idx = int(start_time * fs // segment_length)
            stop_idx = int(stop_time * fs // segment_length)
            labels[start_idx:stop_idx + 1] = label_value
    return labels

def save_preprocessed_data(data, labels, save_path):
    """Save preprocessed EEG data and labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, data=data, labels=labels)

def preprocess_file(edf_path, csv_path, save_dir, target_fs=100, segment_length=2, visualize=False):
    """
    Preprocess a single EDF file.
    :param edf_path: Path to the EDF file.
    :param csv_path: Path to the corresponding CSV annotation file.
    :param save_dir: Directory to save preprocessed data.
    :param target_fs: Target sampling rate.
    :param segment_length: Length of each segment in seconds.
    :param visualize: Whether to visualize spectrograms (default=False).
    """
    print(f"Processing: {edf_path}")
    try:
        data, fs = load_edf_signals(edf_path)
        print(f"Loaded data shape: {data.shape}, Sampling rate: {fs}")
        data = apply_notch_filter(data, fs)
        data = preprocess_signals(data, fs, target_fs, segment_length)
        labels = generate_labels(csv_path, data.shape[0], segment_length, target_fs)
        
        # Save the processed data
        base_name = os.path.splitext(os.path.basename(edf_path))[0]
        save_preprocessed_data(data, labels, os.path.join(save_dir, f"{base_name}.npz"))
        print(f"Preprocessed data saved to {os.path.join(save_dir, f'{base_name}.npz')}")
    except Exception as e:
        print(f"Error processing {edf_path}: {e}")

    """Preprocess a single EDF file."""
    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    data, fs = load_edf_signals(edf_path)
    data = apply_notch_filter(data, fs)
    data = preprocess_signals(data, fs, target_fs, segment_length)
    labels = generate_labels(csv_path, data.shape[0], segment_length, target_fs)
    save_preprocessed_data(data, labels, os.path.join(save_dir, f"{base_name}.npz"))
    print(f"Preprocessed data saved to {os.path.join(save_dir, f'{base_name}.npz')}")

def preprocess_dataset(raw_dir, processed_dir, target_fs=100, segment_length=2):
    """Preprocess all data in a dataset."""
    for split in ["train", "dev", "eval"]:
        raw_split_dir = os.path.join(raw_dir, split)
        processed_split_dir = os.path.join(processed_dir, split)
        for file in os.listdir(raw_split_dir):
            if file.endswith(".edf"):
                edf_path = os.path.join(raw_split_dir, file)
                csv_path = edf_path.replace(".edf", ".csv")
                preprocess_file(edf_path, csv_path, processed_split_dir, target_fs, segment_length)

def split_dataset(source_folder, train_folder, dev_folder, eval_folder, train_ratio=0.7, dev_ratio=0.15, seed=42):
    """Split dataset into train, dev, and eval subsets."""
    np.random.seed(seed)
    all_files = [f for f in os.listdir(source_folder) if f.endswith('.edf')]

    # Shuffle and split the data
    np.random.shuffle(all_files)
    n_total = len(all_files)
    n_train = int(train_ratio * n_total)
    n_dev = int(dev_ratio * n_total)

    train_files = all_files[:n_train]
    dev_files = all_files[n_train:n_train + n_dev]
    eval_files = all_files[n_train + n_dev:]

    # Move files into respective folders
    for file_set, dest_folder in zip([train_files, dev_files, eval_files], 
                                     [train_folder, dev_folder, eval_folder]):
        os.makedirs(dest_folder, exist_ok=True)
        for file_name in file_set:
            src = os.path.join(source_folder, file_name)
            dest = os.path.join(dest_folder, file_name)
            shutil.copy(src, dest)
        print(f"Copied {len(file_set)} files to {dest_folder}")

# Paths
raw_data_dir = "data/raw"
processed_data_dir = "data/processed"

# Preprocess the dataset
preprocess_dataset(raw_data_dir, processed_data_dir)
