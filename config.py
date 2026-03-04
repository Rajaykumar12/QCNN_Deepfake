# Configuration for Quantum-Enhanced CNN with 10,000 Total Image Dataset

## Dataset Configuration
IMG_SIZE = 128
CLASSES = ['real', 'fake']

# Total dataset limit (including train/val/test)
TOTAL_MAX_SAMPLES = 10000
MAX_SAMPLES_PER_CLASS = 5000  # 5K real + 5K fake = 10K total
USE_BALANCED_SAMPLING = True

# Dataset splits (percentages of the 10,000 total)
# Standard split: 80% train, 10% val, 10% test
# Or 70% train, 15% val, 15% test
TRAIN_SPLIT = 0.80  # 8,000 images for training
VAL_SPLIT = 0.10    # 1,000 images for validation  
TEST_SPLIT = 0.10   # 1,000 images for testing

## Training Configuration  
BATCH_SIZE = 16
EPOCHS = 150
VAL_SPLIT = 0.125  # Validation from training set (10% of total becomes ~12.5% of train)
RANDOM_SEED = 42

## Optimization Configuration
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 15

## Quantum Circuit Configuration
N_QUBITS = 4
N_LAYERS = 4

## Directory Configuration (fallbacks)
TRAIN_DIR = "./dataset"
PREPROCESSED_DIR = "./preprocessed"

# IMPORTANT: 10,000 Image Constraint
# =====================================
# Total dataset: 10,000 images (5,000 fake + 5,000 real)
# Typical usage:
#   - Train: 8,000 images (80%)
#   - Val:   1,000 images (10% - from train split or separate)
#   - Test:  1,000 images (10% - separate holdout)
# 
# The code automatically limits loading to TOTAL_MAX_SAMPLES
# and ensures balanced sampling across classes.