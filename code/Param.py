
EMBED_SIZE = 64
DURATION = 8
MALPOS = 8

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.
MODEL_DEPTH = 18

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

# TRAIN
EPOCH = 50
SGD_EPOCH = 100
BATCH_SIZE = 10

# VIDEO
FRAME_WIDTH = 112
FRAME_HEIGHT = 112

# FILE
FEATURE_DIR = ''
SAMPLE_DIR = ''
MAXFOLD = ''
DATA_FOLDS = MAXFOLD+'/datapath.txt'

# FEATURE_DIR = '/media/kayzhou/Seagate Expansion Drive/Feature'
# MAXFOLD = '/media/kayzhou/Seagate Expansion Drive/B data'

# MODEL

