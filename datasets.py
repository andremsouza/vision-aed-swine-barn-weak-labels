"""Dataset for audio files with annotations.

This module contains the WaveformDataset class, which is a dataset for audio files
with annotations. It is used to load audio files and their annotations, and to
generate samples from them.

Example:
    >>> dataset = WaveformDataset(
    ...     annotations_file="30_09_2020_new.csv",
    ...     data_dir="data/audio/",
    ...     num_classes=9,
    ...     sample_seconds=1.0,
    ...     sample_rate=16000,
    ... )
    >>> len(dataset)
    162
    >>> sample, target = dataset[0]
    >>> sample.shape
    torch.Size([16000])
    >>> target.shape
    torch.Size([9])
"""

# %%
from collections import OrderedDict
from datetime import datetime
import os
from typing import Callable
import warnings

import librosa as lr
import numpy as np
import pandas as pd
import torch

from data import extract_data_from_filename

# %%


class WaveformDataset(torch.utils.data.Dataset):
    """Dataset for audio files with annotations.

    Args:
        annotations_file (str | pd.DataFrame): path to annotations file or dataframe
        data_dir (str): path to audio files
        num_classes (int, optional): number of classes. Defaults to 9.
        sample_seconds (float, optional): duration of samples in seconds.
            Defaults to 1.0.
        sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
        transform (Callable, optional): transform to apply to samples. Defaults to None.
        target_transform (Callable, optional): transform to apply to labels.
            Defaults to None.
    """

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        data_dir: str,
        num_classes: int = 9,
        sample_seconds: float = 1.0,
        sample_rate: int = 16000,
        prune_invalid: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            annotations_file (str | pd.DataFrame): path to annotations file or dataframe
            data_dir (str): path to audio files
            num_classes (int, optional): number of classes. Defaults to 9.
            sample_seconds (float, optional): duration of samples in seconds.
                Defaults to 1.0.
            sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
            prune_invalid (bool, optional): whether to prune invalid samples.
                Defaults to False.
            transform (Callable, optional): transform to apply to samples.
                Defaults to None.
            target_transform (Callable, optional): transform to apply to labels.
                Defaults to None.

        Raises:
            TypeError: if annotations_file index is not str or datetime64[ns]

        Examples:
            >>> dataset = WaveformDataset(
            ...     annotations_file="30_09_2020_new.csv",
            ...     data_dir="data/audio/",
            ...     num_classes=9,
            ...     sample_seconds=1.0,
            ...     sample_rate=16000,
            ... )
            >>> len(dataset)
            162
            >>> sample, target = dataset[0]
            >>> sample.shape
            torch.Size([16000])
            >>> target.shape
            torch.Size([9])
        """
        super().__init__()
        if isinstance(annotations_file, str):
            self.annotations: pd.DataFrame = pd.read_csv(annotations_file, index_col=0)
            self.annotations.index = pd.to_datetime(
                self.annotations.index, dayfirst=True
            )
        else:
            self.annotations = annotations_file
        self.data_dir: str = data_dir
        self.num_classes: int = num_classes
        self.sample_seconds: float = sample_seconds
        self.sample_rate: int = sample_rate
        self.prune_invalid: bool = prune_invalid
        self.transform: Callable | None = transform
        self.target_transform: Callable | None = target_transform
        # Find audio files in data_dir
        self.audio_files: list = [
            file
            for file in os.listdir(self.data_dir)
            if file.endswith(".wav") or file.endswith(".mp3")
        ]
        # Get audio file data
        audio_files_data_list: list = []
        for file in self.audio_files:
            try:
                audio_files_data_list.append(extract_data_from_filename(file))
            except AttributeError:
                warnings.warn(f"Could not extract data from {file}")
        self.audio_files_data: pd.DataFrame = pd.DataFrame(audio_files_data_list)
        self.audio_files_data.sort_values(by="timestamp", inplace=True)
        self.data = self._annotations_to_samples()
        if self.prune_invalid:
            self._prune_invalid_samples()

    def _annotations_to_samples(self) -> pd.DataFrame:
        """Transform annotations into samples with start and duration for each file.

        Returns:
            pd.DataFrame: dataframe with samples
        """
        samples: list[OrderedDict[str, object]] | pd.DataFrame = []
        for idx, row in self.annotations.iterrows():
            # Get annotation timestamp
            if self.annotations.index.dtype == "object":
                timestamp: datetime = datetime.strptime(
                    row.name,  # type: ignore
                    "%d/%m/%Y %H:%M:%S",
                )
            elif self.annotations.index.dtype == "datetime64[ns]":
                timestamp = row.name.to_pydatetime()  # type: ignore
            else:
                raise TypeError(
                    f"Timestamp type not recognized at index {idx}."
                    f"Use str or datetime64[ns]."
                )
            # Get audio file
            try:
                audio_file_data = self.audio_files_data.loc[
                    (self.audio_files_data["timestamp"] <= timestamp)
                ].iloc[-1]
                audio_file_timestamp = audio_file_data["timestamp"]
                audio_filename = audio_file_data["filename"]
            except IndexError:
                warnings.warn(f"Could not find audio file for {timestamp}")
                continue
            # get annotation duration in seconds
            duration = row.iloc[0]  # Should be in first column
            # Calculate annotation offset in seconds
            offset: float = (timestamp - audio_file_timestamp).seconds
            # Get audio file path
            audio_file_path: str = os.path.join(self.data_dir, audio_filename)
            # Calculate number of new samples
            num_samples: int = int(duration // self.sample_seconds)
            # Generate and append samples
            for i in range(num_samples):
                # ordered dict to preserve order

                sample = OrderedDict(
                    [
                        ("audio_file_path", audio_file_path),
                        ("offset", offset + i * self.sample_seconds),
                        ("duration", self.sample_seconds),
                    ]
                )
                # last num_classes columns should be the labels
                sample.update(row.iloc[-self.num_classes :].to_dict())
                samples.append(sample)  # type: ignore
        samples = pd.DataFrame(samples)
        return samples

    def _prune_invalid_samples(self) -> None:
        """Prune invalid samples, e.g., with invalid/insufficient size."""
        # Iterate over samples
        # If a sample has invalid size, remove it
        drop_indices: list = []
        # Try to get each sample
        for idx in range(len(self)):  # pylint: disable=consider-using-enumerate
            try:
                # Load audio
                self._load_audio(idx)
            except ValueError:
                # Add idx to list of indices to drop
                drop_indices.append(idx)
        # Drop invalid samples
        self.data.drop(drop_indices, inplace=True)
        # Reset index
        self.data.reset_index(drop=True, inplace=True)
        warnings.warn(f"Pruned {len(drop_indices)} invalid samples.")

    def _load_audio(self, idx: int, normalize: bool = True) -> tuple[np.ndarray, float]:
        """Load audio file at index idx.

        Args:
            idx (int): index of sample
            normalize (bool, optional): whether to normalize audio. Defaults to True.

        Returns:
            tuple[np.ndarray, int]: waveform and sample rate
        """
        # Get sample
        sample_data: pd.Series = self.data.iloc[idx]
        # Get audio file path
        audio_file_path: str = sample_data["audio_file_path"]
        # Get offset and duration
        offset: float = sample_data["offset"]
        duration: float = sample_data["duration"]
        # Load audio file
        waveform, sample_rate = lr.load(
            path=audio_file_path, sr=self.sample_rate, offset=offset, duration=duration
        )
        assert sample_rate == self.sample_rate
        if waveform.shape[0] != self.sample_rate * self.sample_seconds:
            raise ValueError(
                f"Sample at index {idx} has shape {waveform.shape}. "
                f"Expected shape is {(self.sample_rate * self.sample_seconds,)}"
            )
        # Normalize waveform to [-1, 1]
        if normalize:
            waveform = lr.util.normalize(waveform)
        return waveform, sample_rate

    def __len__(self) -> int:
        """Get number of samples in dataset.

        Returns:
            int: number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and label at index idx.

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sample and label
        """
        # Get sample data
        sample_data: pd.Series = self.data.iloc[idx]
        # Load audio
        try:
            waveform, _ = self._load_audio(idx)
        except ValueError:
            warnings.warn(f"Sample at index {idx} could not be loaded.")
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        sample = torch.from_numpy(waveform).float()
        # Get label
        target = torch.tensor(
            sample_data.iloc[-self.num_classes :].to_numpy().astype(float)
        )
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


class SpectrogramDataset(WaveformDataset):
    """Dataset for audio files with annotations.

    Args:
        annotations_file (str | pd.DataFrame): path to annotations file or dataframe
        data_dir (str): path to audio files
        num_classes (int, optional): number of classes. Defaults to 9.
        sample_seconds (float, optional): duration of samples in seconds.
            Defaults to 1.0.
        sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
        prune_invalid (bool, optional): whether to prune invalid samples.
            Defaults to False.
        window_length_seconds (float, optional): window length in seconds.
            Defaults to 0.025.
        hop_length_seconds (float, optional): hop length in seconds.
            Defaults to 0.01.
        num_mel_filters (int, optional): number of mel filters. Defaults to 64.
        mel_filterbank_min_freq (float, optional): minimum frequency for mel filterbank.
            Defaults to 125.0.
        mel_filterbank_max_freq (float, optional): maximum frequency for mel filterbank.
            Defaults to 7500.0.
        log_offset (float, optional): offset for log-mel spectrogram. Defaults to 0.01.
        example_window_length_seconds (float, optional): window length for spectrogram
            examples in seconds. Defaults to 0.96.
        example_hop_length_seconds (float, optional): hop length for spectrogram
            examples in seconds. Defaults to 0.96.
        transform (Callable, optional): transform to apply to samples.
            Defaults to None.
        target_transform (Callable, optional): transform to apply to labels.
            Defaults to None.
    """

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        data_dir: str,
        num_classes: int = 9,
        sample_seconds: float = 1.0,
        sample_rate: int = 16000,
        prune_invalid: bool = False,
        window_length_seconds: float = 0.025,
        hop_length_seconds: float = 0.01,
        num_mel_filters: int = 64,
        mel_filterbank_min_freq: float = 125.0,
        mel_filterbank_max_freq: float = 7500.0,
        log_offset: float = 0.01,
        example_window_length_seconds: float = 0.96,
        example_hop_length_seconds: float = 0.96,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
        annotations_file (str | pd.DataFrame): path to annotations file or dataframe
        data_dir (str): path to audio files
        num_classes (int, optional): number of classes. Defaults to 9.
        sample_seconds (float, optional): duration of samples in seconds.
            Defaults to 1.0.
        sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
        prune_invalid (bool, optional): whether to prune invalid samples.
            Defaults to False.
        window_length_seconds (float, optional): window length in seconds.
            Defaults to 0.025.
        hop_length_seconds (float, optional): hop length in seconds.
            Defaults to 0.01.
        num_mel_filters (int, optional): number of mel filters. Defaults to 64.
        mel_filterbank_min_freq (float, optional): minimum frequency for mel filterbank.
            Defaults to 125.0.
        mel_filterbank_max_freq (float, optional): maximum frequency for mel filterbank.
            Defaults to 7500.0.
        log_offset (float, optional): offset for log-mel spectrogram. Defaults to 0.01.
        example_window_length_seconds (float, optional): window length for spectrogram
            examples in seconds. Defaults to 0.96.
        example_hop_length_seconds (float, optional): hop length for spectrogram
            examples in seconds. Defaults to 0.96.
        transform (Callable, optional): transform to apply to samples.
            Defaults to None.
        target_transform (Callable, optional): transform to apply to labels.
            Defaults to None.
        """
        super().__init__(
            annotations_file=annotations_file,
            data_dir=data_dir,
            num_classes=num_classes,
            sample_seconds=sample_seconds,
            sample_rate=sample_rate,
            prune_invalid=prune_invalid,
            transform=transform,
            target_transform=target_transform,
        )
        self.window_length_seconds: float = window_length_seconds
        self.hop_length_seconds: float = hop_length_seconds
        self.num_mel_filters: int = num_mel_filters
        self.mel_filterbank_min_freq: float = mel_filterbank_min_freq
        self.mel_filterbank_max_freq: float = mel_filterbank_max_freq
        self.log_offset: float = log_offset
        self.example_window_length_seconds: float = example_window_length_seconds
        self.example_hop_length_seconds: float = example_hop_length_seconds
        # Compute spectrogram parameters
        self.window_length_samples = int(
            np.round(self.window_length_seconds * self.sample_rate)
        )
        self.hop_length_samples = int(
            np.round(self.hop_length_seconds * self.sample_rate)
        )
        self.fft_length_samples = 2 ** int(
            np.ceil(np.log(self.window_length_samples) / np.log(2.0))
        )
        self.spectrogram_sample_rate = 1.0 / self.hop_length_seconds
        self.example_window_length_samples = int(
            np.round(self.example_window_length_seconds * self.spectrogram_sample_rate)
        )
        self.example_hop_length_samples = int(
            np.round(self.example_hop_length_seconds * self.spectrogram_sample_rate)
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and label at index idx.

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sample and label
        """
        # Get sample data
        sample_data: pd.Series = self.data.iloc[idx]
        # Load audio
        try:
            waveform, _ = self._load_audio(idx)
        except ValueError:
            warnings.warn(f"Sample at index {idx} could not be loaded.")
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        # Compute log-mel spectrogram
        stft = lr.stft(
            y=waveform,
            n_fft=self.fft_length_samples,
            hop_length=self.hop_length_samples,
            win_length=self.window_length_samples,
        )
        mel_filterbank = lr.filters.mel(
            sr=self.sample_rate,
            n_fft=self.fft_length_samples,
            n_mels=self.num_mel_filters,
            fmin=self.mel_filterbank_min_freq,
            fmax=self.mel_filterbank_max_freq,
        )
        mel_spectrogram = np.dot(mel_filterbank, np.abs(stft))
        log_mel_spectrogram = np.log(mel_spectrogram + self.log_offset)
        # Frame spectrogram into sample
        log_mel_examples = lr.util.frame(
            log_mel_spectrogram,
            frame_length=self.example_window_length_samples,
            hop_length=self.example_hop_length_samples,
        )
        # Swap axes to make patches the first dimension
        log_mel_examples = np.transpose(log_mel_examples, axes=(2, 0, 1))
        sample = torch.from_numpy(log_mel_examples.copy()).float()
        sample = sample.squeeze()
        # Get label
        target = torch.tensor(
            sample_data.iloc[-self.num_classes :].to_numpy().astype(float)
        )
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


class MFCCDataset(SpectrogramDataset):
    """Dataset for audio files with annotations.

    Args:
        annotations_file (str | pd.DataFrame): path to annotations file or dataframe
        data_dir (str): path to audio files
        num_classes (int, optional): number of classes. Defaults to 9.
        sample_seconds (float, optional): duration of samples in seconds.
            Defaults to 1.0.
        sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
        prune_invalid (bool, optional): whether to prune invalid samples.
            Defaults to False.
        num_mfcc (int, optional): number of MFCCs. Defaults to 20.
        window_length_seconds (float, optional): window length in seconds.
            Defaults to 0.025.
        hop_length_seconds (float, optional): hop length in seconds.
            Defaults to 0.01.
        num_mel_filters (int, optional): number of mel filters. Defaults to 64.
        mel_filterbank_min_freq (float, optional): minimum frequency for mel filterbank.
            Defaults to 125.0.
        mel_filterbank_max_freq (float, optional): maximum frequency for mel filterbank.
            Defaults to 7500.0.
        log_offset (float, optional): offset for log-mel spectrogram. Defaults to 0.01.
        example_window_length_seconds (float, optional): window length for spectrogram
            examples in seconds. Defaults to 0.96.
        example_hop_length_seconds (float, optional): hop length for spectrogram
            examples in seconds. Defaults to 0.96.
        transform (Callable, optional): transform to apply to samples.
            Defaults to None.
        target_transform (Callable, optional): transform to apply to labels.
            Defaults to None.
    """

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        data_dir: str,
        num_classes: int = 9,
        sample_seconds: float = 1.0,
        sample_rate: int = 16000,
        prune_invalid: bool = False,
        num_mfcc: int = 20,
        window_length_seconds: float = 0.025,
        hop_length_seconds: float = 0.01,
        num_mel_filters: int = 64,
        mel_filterbank_min_freq: float = 125.0,
        mel_filterbank_max_freq: float = 7500.0,
        log_offset: float = 0.01,
        example_window_length_seconds: float = 0.96,
        example_hop_length_seconds: float = 0.96,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
        annotations_file (str | pd.DataFrame): path to annotations file or dataframe
        data_dir (str): path to audio files
        num_classes (int, optional): number of classes. Defaults to 9.
        sample_seconds (float, optional): duration of samples in seconds.
            Defaults to 1.0.
        sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
        prune_invalid (bool, optional): whether to prune invalid samples.
            Defaults to False.
        num_mfcc (int, optional): number of MFCCs. Defaults to 20.
        window_length_seconds (float, optional): window length in seconds.
            Defaults to 0.025.
        hop_length_seconds (float, optional): hop length in seconds.
            Defaults to 0.01.
        num_mel_filters (int, optional): number of mel filters. Defaults to 64.
        mel_filterbank_min_freq (float, optional): minimum frequency for mel filterbank.
            Defaults to 125.0.
        mel_filterbank_max_freq (float, optional): maximum frequency for mel filterbank.
            Defaults to 7500.0.
        log_offset (float, optional): offset for log-mel spectrogram. Defaults to 0.01.
        example_window_length_seconds (float, optional): window length for spectrogram
            examples in seconds. Defaults to 0.96.
        example_hop_length_seconds (float, optional): hop length for spectrogram
            examples in seconds. Defaults to 0.96.
        transform (Callable, optional): transform to apply to samples.
            Defaults to None.
        target_transform (Callable, optional): transform to apply to labels.
            Defaults to None.
        """
        super().__init__(
            annotations_file=annotations_file,
            data_dir=data_dir,
            num_classes=num_classes,
            sample_seconds=sample_seconds,
            sample_rate=sample_rate,
            prune_invalid=prune_invalid,
            window_length_seconds=window_length_seconds,
            hop_length_seconds=hop_length_seconds,
            num_mel_filters=num_mel_filters,
            mel_filterbank_min_freq=mel_filterbank_min_freq,
            mel_filterbank_max_freq=mel_filterbank_max_freq,
            log_offset=log_offset,
            example_window_length_seconds=example_window_length_seconds,
            example_hop_length_seconds=example_hop_length_seconds,
            transform=transform,
            target_transform=target_transform,
        )
        self.num_mfcc = num_mfcc

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and label at index idx.

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sample and label
        """
        # Get sample data
        sample_data: pd.Series = self.data.iloc[idx]
        # Load audio
        # Load audio
        try:
            waveform, _ = self._load_audio(idx)
        except ValueError:
            warnings.warn(f"Sample at index {idx} could not be loaded.")
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        # Compute MFCCs
        mfccs = lr.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.num_mfcc,
            n_fft=self.fft_length_samples,
            hop_length=self.hop_length_samples,
            win_length=self.window_length_samples,
            n_mels=self.num_mel_filters,
            fmin=self.mel_filterbank_min_freq,
            fmax=self.mel_filterbank_max_freq,
        )
        # Frame MFCCs into sample
        mfcc_examples = lr.util.frame(
            mfccs,
            frame_length=self.example_window_length_samples,
            hop_length=self.example_hop_length_samples,
        )
        # Swap axes to make patches the first dimension
        mfcc_examples = np.transpose(mfcc_examples, axes=(2, 0, 1))
        sample = torch.from_numpy(mfcc_examples.copy()).float()
        sample = sample.squeeze()
        # Get label
        target = torch.tensor(
            sample_data.iloc[-self.num_classes :].to_numpy().astype(float)
        )
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


# %%
# Test dataset
if __name__ == "__main__":
    waveform_dataset = WaveformDataset(
        annotations_file="30_09_2020_new.csv",
        data_dir="data/audio/",
        num_classes=9,
        sample_seconds=1.0,
        sample_rate=16000,
    )
    waveform_dataset_pruned = WaveformDataset(
        annotations_file="30_09_2020_new.csv",
        data_dir="data/audio/",
        num_classes=9,
        sample_seconds=1.0,
        sample_rate=16000,
        prune_invalid=True,
    )
    spectrogram_dataset = SpectrogramDataset(
        annotations_file="30_09_2020_new.csv",
        data_dir="data/audio/",
        num_classes=9,
        sample_seconds=1.0,
        sample_rate=16000,
        prune_invalid=True,
    )
    mfcc_dataset = MFCCDataset(
        annotations_file="30_09_2020_new.csv",
        data_dir="data/audio/",
        num_classes=9,
        sample_seconds=1.0,
        sample_rate=16000,
        prune_invalid=True,
    )
    # Iterate over datasets
    NUM_ITERATIONS = 5
    # Measure times for dataset
    print("Measuring times...")
    times: list = []
    for i in range(NUM_ITERATIONS):
        start = datetime.now()
        for sample, label in waveform_dataset:  # type: ignore
            pass
        end = datetime.now()
        times.append((end - start).total_seconds())
    # Measure times for pruned dataset
    print("Measuring times for pruned dataset...")
    times_pruned: list = []
    for i in range(NUM_ITERATIONS):
        start = datetime.now()
        for sample, label in waveform_dataset_pruned:  # type: ignore
            pass
        end = datetime.now()
        times_pruned.append((end - start).total_seconds())
    print("Measuring times for spectrogram dataset...")
    times_spectrogram: list = []
    for i in range(NUM_ITERATIONS):
        start = datetime.now()
        for sample, label in spectrogram_dataset:  # type: ignore
            pass
        end = datetime.now()
        times_spectrogram.append((end - start).total_seconds())
    print("Measuring times for MFCC dataset...")
    times_mfcc: list = []
    for i in range(NUM_ITERATIONS):
        start = datetime.now()
        for sample, label in mfcc_dataset:  # type: ignore
            pass
        end = datetime.now()
        times_mfcc.append((end - start).total_seconds())
    # Print results
    # Without pruning
    print(f"Dataset: {np.mean(times)}s")
    # With pruning
    print(f"Pruned dataset: {np.mean(times_pruned)}s")
    # Spectrogram dataset
    print(f"Spectrogram dataset: {np.mean(times_spectrogram)}s")
    # MFCC dataset
    print(f"MFCC dataset: {np.mean(times_mfcc)}s")
    # Pruning factor
    print(f"Pruned dataset is {np.mean(times)/np.mean(times_pruned)} times faster.")
    # Spectrogram overhead
    print(
        f"Spectrogram dataset is {np.mean(times_pruned)/np.mean(times_spectrogram)} "
        f"times faster."
    )
    # MFCC overhead
    print(
        f"MFCC dataset is {np.mean(times_pruned)/np.mean(times_mfcc)} " f"times faster."
    )
    # Measure times for spectrogram dataset
    print("Done.")


# %%
