import torch
import os
import random
import numpy as np

NUM_CLASSES = 10
IMAGE_SIZE = 64  
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

