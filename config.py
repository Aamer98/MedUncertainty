# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 500

# Dataset
DATA_DIR = "/home/as26840@ens.ad.etsmtl.ca/data/plightning_toy"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0. 1]
PRECISION = 16
