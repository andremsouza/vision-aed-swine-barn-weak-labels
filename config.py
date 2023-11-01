"""Configuration file for the project.

This file contains all the global constants used in the project.
"""
# %%
import torch

# %% [markdown]
# # Constants

# %%
# ANNOTATION_FILE = "/srv/andre/30_09_2020.csv"
# ANNOTATION_FILE = "./30_09_2020_cropped.csv"
ANNOTATION_FILE: str = "30_09_2020_new.csv"
TRAIN_ANNOTATION_FILE = "./train_annotation.csv"
VAL_ANNOTATION_FILE = "./val_annotation.csv"
TEST_ANNOTATION_FILE = "./test_annotation.csv"

NUM_CLASSES: int = 9

ANNOTATION_SECONDS = 5

FEATURE_FILE = "./features_2020-09-30.csv"
CHUNK_SIZE = 10**3

# DATA_DIRECTORY = "/srv/andre/gbdi_vm/fmvz/10_SWINE_ICMC/20201013/ALA_E/"
DATA_DIRECTORY = "./data/audio/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIRECTORY = "./models/"

BATCH_SIZE = 64
N_MFCCS = 20

PRED_THRESHOLD = 0.5

RANDOM_SEED = 42

SKIP_TRAINED_MODELS = True
FC_BEST_MODELS_ONLY = True

LOG_DIRECTORY = "./logs/"

EARLY_STOPPING_PATIENCE = 16

NUM_WORKERS = 24

USE_PRETRAINED: bool = False


# %% [markdown]
# # VGGIsh Parameters

# %%
# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = "pca_eigen_vectors"
PCA_MEANS_NAME = "pca_means"
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = "vggish/input_features"
INPUT_TENSOR_NAME = INPUT_OP_NAME + ":0"
OUTPUT_OP_NAME = "vggish/embedding"
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ":0"
AUDIO_EMBEDDING_FEATURE_NAME = "audio_embedding"

# %%
