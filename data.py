"""Functions for loading and preprocessing data with PyTorch.

Example:
    >>> from data import load_data
    >>> train_loader, test_loader = load_data(batch_size=32)

"""
# TODO: Adapt for active learning

# %% [markdown]
# # Imports

# %%
import datetime
import os
import re
import warnings

import librosa as lr
import numpy as np
import pandas as pd
import torch

import config

# %% [markdown]
# # Classes

# %%


class AudioDataset(torch.utils.data.Dataset):
    """Audio dataset."""

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        audio_dir: str,
        transform=None,
        target_transform=None,
    ):
        """Initialize dataset.

        Args:
            annotations_file (str): path to annotations file
            audio_dir (str): path to audio directory
            transform (callable, optional): optional transform to be applied on a sample.
                Defaults to None.
            target_transform (callable, optional): optional transform to be applied on
                the target. Defaults to None.
        """
        if isinstance(annotations_file, str):
            self.annotations = pd.read_csv(annotations_file, index_col=0)
            self.annotations.index = pd.to_datetime(self.annotations.index)
        else:
            self.annotations = annotations_file
        self.audio_dir = audio_dir
        # Only .mp4 fiules
        self.audio_list = [
            file for file in os.listdir(self.audio_dir) if file.endswith(".mp4")
        ]
        self.audio_file_data: list | pd.DataFrame = []
        for file in self.audio_list:
            try:
                self.audio_file_data.append(extract_data_from_filename(file))
            except AttributeError:
                pass
        self.audio_file_data = pd.DataFrame(self.audio_file_data)
        self.audio_file_data = self.audio_file_data.sort_values(
            by="timestamp",
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return length of dataset.

        Returns:
            int: length of dataset
        """
        return len(self.annotations) * config.ANNOTATION_SECONDS

    def __getitem__(self, idx):
        """Return sample from dataset.

        Args:
            idx (int): index of sample

        Returns:
            tuple: sample and target
        """
        # Get timestamp
        if self.annotations.index.dtype == "object":
            timestamp = datetime.datetime.strptime(
                self.annotations.iloc[idx // config.ANNOTATION_SECONDS, :].name,
                "%d/%m/%Y %H:%M:%S",
            )
        elif self.annotations.index.dtype == "datetime64[ns]":
            timestamp = self.annotations.iloc[
                idx // config.ANNOTATION_SECONDS, :
            ].name.to_pydatetime()
        else:
            raise TypeError("Timestamp type not recognized")
        # Get audio file
        audio_file = self.audio_file_data.loc[
            (self.audio_file_data["timestamp"] <= timestamp)
        ].iloc[-1]["filename"]
        # Calculate offset in seconds
        offset: float = (
            timestamp
            - self.audio_file_data.loc[
                (self.audio_file_data["timestamp"] <= timestamp)
            ].iloc[-1]["timestamp"]
        ).seconds
        # Get audio file path
        audio_file_path = os.path.join(self.audio_dir, audio_file)
        # Get audio file sample rate
        audio_file_sample_rate = lr.get_samplerate(audio_file_path)
        # Get audio file features
        try:
            audio_file_features = extract_audio_features_from_file(
                audio_file_path,
                audio_file_sample_rate,
                offset + (idx % 5),
                duration=1.0,
                verbose=False,
            )
        except ValueError:
            warnings.warn(
                f"Could not extract features from {audio_file_path} at {offset}."
            )
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        # Match annotation timestamp with audio file timestamp
        patch = match_patch_annotation(
            audio_file_features,
            self.annotations.iloc[idx // config.ANNOTATION_SECONDS, :],
        ).dropna(axis=0)

        # Check if item has enough data
        if patch.shape[0] < 96:
            return self.__getitem__(idx + 1)

        # Separate sample and target, convert to pytorch tensors
        sample = torch.tensor(patch.iloc[:, 5:-9].values).float()
        # Target is the mode of the 9 last columns
        target = torch.tensor(
            patch.iloc[:, -9:].mode(axis=0).astype(float).values
        ).float()

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


# %% [markdown]
# # Functions

# %%


def extract_data_from_filename(filename: str) -> dict:
    """Extract datetime and other data from filename.

    Args:
        filename (str): filename to extract data from

    Returns:
        dict: dictionary with extracted data
    """
    pattern: str = (
        r"^.*ALA_(\w)"
        + r"\)?_(\d)"
        + r"_(\d{4})-(\d{2})-(\d{2})"
        + r"_(\d{2})-(\d{2})-(\d{2}).*$"
    )
    match: re.Match[str] = re.fullmatch(pattern, filename)  # type: ignore
    data = {
        "timestamp": datetime.datetime(
            year=int(match.groups()[2]),
            month=int(match.groups()[3]),
            day=int(match.groups()[4]),
            hour=int(match.groups()[5]),
            minute=int(match.groups()[6]),
            second=int(match.groups()[7]),
        ),
        "filename": filename,
        "ala": match.groups()[0],  # ALA
        "grupo": int(match.groups()[1]),  # GRUPO
    }
    return data


def extract_audio_features_from_file(
    file_path: str,
    file_sample_rate: int,
    offset: float = 0.0,
    duration: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract features from audio file.

    Args:
        file_path (str): path to audio file
        file_sample_rate (int): sample rate of audio file
        verbose (bool, optional): print info. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with extracted features
    """
    if verbose:
        print("File:", file_path)

    if verbose:
        print("0.Extracting info from filename...")
    timestamp = extract_data_from_filename(file_path)["timestamp"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waveform, sample_rate = lr.load(
            file_path, sr=file_sample_rate, offset=offset, duration=duration
        )
    # Check if waveform has enough data
    if waveform.shape[0] < config.SAMPLE_RATE * duration:
        raise ValueError("Not enough data in audio file")
    return extract_audio_features(
        waveform, sample_rate, timestamp + datetime.timedelta(seconds=offset), verbose
    )


def extract_audio_features(
    waveform: np.ndarray,
    sample_rate: int,
    timestamp: datetime.datetime,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract features from audio file.

    Args:
        waveform (np.ndarray): waveform of audio file
        sample_rate (int): sample rate of audio file
        timestamp (datetime.datetime): timestamp of audio file
        verbose (bool, optional): print info. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with extracted features
    """
    number_of_mfcc = config.N_MFCCS
    # Preprocess audio file
    if verbose:
        print(
            f"1.Importing file with librosa (w/ normalize and resample to {config.SAMPLE_RATE}Hz)"
        )
    # Normalize audio
    waveform = lr.util.normalize(waveform)
    # Resample audio as in VGGish
    waveform = lr.resample(
        y=waveform, orig_sr=sample_rate, target_sr=config.SAMPLE_RATE
    )

    if verbose:
        print("2.Log-mel spectrogram patches...")
    try:
        # Compute log-mel spectrogram
        windows_length_samples = int(
            config.STFT_WINDOW_LENGTH_SECONDS * config.SAMPLE_RATE
        )
        hop_length_samples = int(config.STFT_HOP_LENGTH_SECONDS * config.SAMPLE_RATE)
        fft_length = 2 ** int(np.ceil(np.log(windows_length_samples) / np.log(2.0)))
        stft = lr.stft(
            y=waveform,
            n_fft=fft_length,
            hop_length=hop_length_samples,
            win_length=windows_length_samples,
        )
        mel_filterbank = lr.filters.mel(
            sr=config.SAMPLE_RATE,
            n_fft=fft_length,
            n_mels=config.NUM_BANDS,
            fmin=config.MEL_MIN_HZ,
            fmax=config.MEL_MAX_HZ,
        )
        mel_spectrogram = np.dot(mel_filterbank, np.abs(stft))
        log_mel_spectrogram = np.log(mel_spectrogram + config.LOG_OFFSET)

        # Frame spectrogram into patches
        spectrogram_hop_length_samples = int(
            np.round(config.SAMPLE_RATE * config.STFT_HOP_LENGTH_SECONDS)
        )
        spectrogram_sample_rate = config.SAMPLE_RATE / spectrogram_hop_length_samples
        patch_window_length_samples = int(
            np.round(config.EXAMPLE_WINDOW_SECONDS * spectrogram_sample_rate)
        )
        patch_hop_length_samples = int(
            np.round(config.EXAMPLE_HOP_SECONDS * spectrogram_sample_rate)
        )
        log_mel_spectrogram_patches = lr.util.frame(
            log_mel_spectrogram,
            frame_length=patch_window_length_samples,
            hop_length=patch_hop_length_samples,
        )
        # Swap axes to make patches the first dimension
        log_mel_spectrogram_patches = np.transpose(
            log_mel_spectrogram_patches, axes=[2, 0, 1]
        )
    except Exception as exception:  # pylint: disable=broad-except
        print(exception)
        return None

    if verbose:
        print("3.MFCCs...")
    try:
        # Compute MFCCs
        mfccs: list | np.ndarray = []
        for i in range(log_mel_spectrogram_patches.shape[0]):
            mfccs.append(  # type: ignore
                lr.feature.mfcc(
                    S=log_mel_spectrogram_patches[i],
                    n_mfcc=number_of_mfcc,
                )
            )
        mfccs = np.array(mfccs)
        # Compute delta MFCCs
        delta_mfccs: list | np.ndarray = []
        for i in range(log_mel_spectrogram_patches.shape[0]):
            delta_mfccs.append(lr.feature.delta(mfccs[i], order=1))  # type: ignore
        delta_mfccs = np.array(delta_mfccs)
        # Compute delta-delta MFCCs
        delta_delta_mfccs: list | np.ndarray = []
        for i in range(log_mel_spectrogram_patches.shape[0]):
            delta_delta_mfccs.append(lr.feature.delta(mfccs[i], order=2))  # type: ignore
        delta_delta_mfccs = np.array(delta_delta_mfccs)
    except Exception as exception:  # pylint: disable=broad-except
        print(exception)
        return None

    # Build rows of the dataframe
    if verbose:
        print("4.Building dataframe...")

    # Get number of patches
    n_patches = log_mel_spectrogram_patches.shape[0]

    # Get number of seconds per patch
    n_seconds_per_patch = config.EXAMPLE_HOP_SECONDS

    # Get seconds per times index per patch
    seconds_per_times_index_per_patch = config.EXAMPLE_HOP_SECONDS / config.NUM_FRAMES

    rows = []
    for i in range(n_patches):
        for j in range(config.NUM_FRAMES):
            row = {
                "file_timestamp": timestamp,
                "patch": i,
                "patch_frame": j,
                "patch_timestamp": (
                    timestamp
                    + datetime.timedelta(
                        seconds=i * n_seconds_per_patch
                        + j * seconds_per_times_index_per_patch
                    )
                ),
                "patch_timestamp_seconds": (
                    i * n_seconds_per_patch + j * seconds_per_times_index_per_patch
                ),
            }
            # Expand spectrogram bands to columns
            for k in range(config.NUM_BANDS):
                row[f"band_{k}"] = log_mel_spectrogram_patches[i][k][j]
            # Expand MFCCs to columns
            for k in range(number_of_mfcc):
                row[f"mfcc_{k}"] = mfccs[i][k][j]
                row[f"delta_mfcc_{k}"] = delta_mfccs[i][k][j]
                row[f"delta_delta_mfcc_{k}"] = delta_delta_mfccs[i][k][j]
            rows.append(row)

    if verbose:
        print("DONE:", timestamp)

    return pd.DataFrame(rows)


def match_patch_annotation(
    patches: pd.DataFrame, annotations: pd.Series | pd.DataFrame, inplace: bool = False
) -> None | pd.DataFrame:
    """Match patch annotations by timestamp.

    Args:
        patches (pd.DataFrame): Patches dataframe.
        annotations (pd.DataFrame): Annotations dataframe.
        inplace (bool, optional): Whether to modify the patches dataframe in place.
            Defaults to False.

    Returns:
        pd.DataFrame: Patches dataframe with annotations. None if inplace is True.
    """
    # If the annotations are a series, convert to dataframe
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T
        annotations.index = pd.to_datetime(annotations.index)
    if not inplace:
        patches = patches.copy()
    for index, row in patches.iterrows():
        # Get timestamp
        timestamp = pd.to_datetime(row["patch_timestamp"]).strftime(
            "%d/%m/%Y %H:%M:%S.%f"
        )
        timestamp_ns_before = (
            pd.to_datetime(row["patch_timestamp"])
            - pd.Timedelta(seconds=config.ANNOTATION_SECONDS - 1, milliseconds=999)
        ).strftime("%d/%m/%Y %H:%M:%S.%f")
        # Find the closest annnotation  less than N seconds
        annotation = annotations.loc[timestamp_ns_before:timestamp]
        # If there is no annotation within N seconds, skip
        if annotation.empty:
            continue
        # concatenate the annotation to the data frame
        for col in annotations.columns:
            patches.loc[index, col] = annotation[col].tail(n=1).values[0]
    if not inplace:
        return patches
    return None


# %% [markdown]
# # Test

# %%
if __name__ == "__main__":
    dataset = AudioDataset(
        annotations_file=config.ANNOTATION_FILE,
        audio_dir=config.DATA_DIRECTORY,
        transform=lambda x: x[:, :64],
    )
    item = dataset[0]
    print(item)
    print(item[0].shape)
    print(item[1].shape)
# %%
