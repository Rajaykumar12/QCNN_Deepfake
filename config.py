# Configuration for Minimal Quantum CNN — Deepfake Detection
# Designed for ~10,000 image datasets

## Dataset
CLASSES = ['Real', 'Fake']         # Folder names (case-sensitive)
IMG_SIZE = 64                     # Input image resolution
MAX_SAMPLES_PER_CLASS = 5000      # Cap per class per split (quantum is slow on 70k)
RANDOM_SEED = 42

# The dataset already provides Train / Validation / Test splits.
# These fractions are only used if you load a flat (unsplit) directory.
VAL_SPLIT  = 0.10
TEST_SPLIT = 0.10

## Training
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15

## Quantum circuit
# 4 qubits → 4-channel output (one qubit per pixel of the 2×2 patch)
# More expressive: each qubit carries info about one pixel, entanglement
# lets the circuit learn correlations between all 4 pixels.
N_QUBITS = 4
N_LAYERS = 2                     # Variational depth (2 layers reduces barren plateau risk vs 3)

## Directories
DATASET_DIR  = "archive/Dataset"   # contains Train/ Validation/ Test/ sub-dirs
OUTPUT_DIR          = "./minimal_quantum_results"
HYBRID_OUTPUT_DIR   = "./hybrid_quantum_results"
PRETRAIN_OUTPUT_DIR = "./pretrained_quantum"       # Phase 1 output
CACHED_FEATURES_DIR = "./cached_quantum_features"  # Phase 2 output
PHASE3_OUTPUT_DIR   = "./phase3_results"           # Phase 3 output