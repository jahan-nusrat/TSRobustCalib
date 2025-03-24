import os
from tusz_utilities import tusz_to_segment_mat

def preprocess_train_dev_eval(
    root_data_path: str,
    root_save_path: str,
    resample_freq: float,
    segment_len: int,
    spectrogram_args: dict,
    notch_args: dict
):

    #Preprocess train, dev, and eval sets separately to avoid data leakage.
    # 1) Train
    train_path = os.path.join(root_data_path, "train")
    train_save = os.path.join(root_save_path, "train")
    os.makedirs(train_save, exist_ok=True)
    tusz_to_segment_mat(
        data_path=train_path,
        save_path=train_save,
        resample_freq=resample_freq,
        segment_len=segment_len,
        spectrogram_args=spectrogram_args,
        notch_args=notch_args
    )
    # 2) Dev
    dev_path = os.path.join(root_data_path, "dev")
    dev_save = os.path.join(root_save_path, "dev")
    os.makedirs(dev_save, exist_ok=True)
    tusz_to_segment_mat(
        data_path=dev_path,
        save_path=dev_save,
        resample_freq=resample_freq,
        segment_len=segment_len,
        spectrogram_args=spectrogram_args,
        notch_args=notch_args
    )
    # 3) Eval
    eval_path = os.path.join(root_data_path, "eval")
    eval_save = os.path.join(root_save_path, "eval")
    os.makedirs(eval_save, exist_ok=True)
    tusz_to_segment_mat(
        data_path=eval_path,
        save_path=eval_save,
        resample_freq=resample_freq,
        segment_len=segment_len,
        spectrogram_args=spectrogram_args,
        notch_args=notch_args
    )


def main():
    root_data_path = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Raw_data"
    root_save_path = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Processed_data"

    preprocess_train_dev_eval(
        root_data_path=root_data_path,
        root_save_path=root_save_path,
        resample_freq=200.0,
        segment_len=1024,
        spectrogram_args={"nperseg": 256, "noverlap": 128},
        notch_args={"w0": 60.0, "Q": 30.0}
    )

if __name__ == "__main__":
    main()
