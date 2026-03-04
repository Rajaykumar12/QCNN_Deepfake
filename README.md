# Quantum-Enhanced CNN for Deepfake Detection

End-to-end quantum-enhanced CNN for deepfake detection. The quantum layer parameters update via backpropagation alongside the CNN weights.

## Setup

**Requires Python 3.11 or 3.12** (TensorFlow has no Python 3.13+ wheels yet)

```bash
# Create conda environment
conda create -n qcnn python=3.12 -y
conda activate qcnn

# Install dependencies
pip install -r requirements.txt
```

## Architecture

```
Input: 64×64×1 grayscale
    ↓
MinimalQuantumLayer (trainable, 2 qubits × 2 layers)
  • Extract 2×2 patches → 32×32 = 1024 patches per image
  • Encode 4 pixel values via RX·RY gates
  • CNOT entanglement + trainable variational layers
  • Measure ⟨Z₀⟩, ⟨Z₁⟩ → 2 channels
    ↓
Conv2D(32) → BN → Pool → Dropout
Conv2D(64) → BN → Pool → Dropout  
Conv2D(128) → BN → GAP → Dropout → Dense(1, sigmoid)
    ↓
Output: probability [0, 1] (0 = Real, 1 = Fake)
```

## Dataset Structure

```
archive/Dataset/
├── Train/
│   ├── Real/
│   └── Fake/
├── Validation/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

## Usage

```bash
# Train
python train_minimal_quantum.py --dataset_dir archive/Dataset

# With custom settings
python train_minimal_quantum.py --dataset_dir archive/Dataset \
    --max_samples 3000 --epochs 30 --batch_size 16

# Single image inference
python inference_minimal_quantum.py \
    --model_path ./minimal_quantum_results/best_model.keras \
    --image_path ./test_image.jpg

# Batch inference
python inference_minimal_quantum.py \
    --model_path ./minimal_quantum_results/best_model.keras \
    --dataset_dir ./test_dataset \
    --output_csv results.csv
```

## Configuration

Edit `config.py` to change defaults:

| Setting | Default | Notes |
|---------|---------|-------|
| `IMG_SIZE` | 64 | Input resolution |
| `N_QUBITS` | 2 | Qubits in quantum circuit |
| `N_LAYERS` | 2 | Variational layers |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 50 | Max epochs |
| `MAX_SAMPLES_PER_CLASS` | 5000 | Cap per class (quantum is slow) |

## Output

After training, `minimal_quantum_results/` contains:

| File | Description |
|------|-------------|
| `best_model.keras` | Best val-accuracy checkpoint |
| `final_model.keras` | Model at end of training |
| `history.npy` | Training history |
| `training_history.png` | Accuracy/loss curves |
| `test_results.txt` | Test-set metrics |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM | `--batch_size 8` |
| Training too slow | `--max_samples 2000` |
| Overfitting | Increase `--max_samples`, already has BatchNorm + Dropout + L2 |
| PennyLane errors | `pip install pennylane>=0.44.0` |

## Files

| File | Purpose |
|------|---------|
| `config.py` | Central configuration |
| `minimal_quantum_cnn.py` | Quantum layer + model |
| `train_minimal_quantum.py` | Training script |
| `inference_minimal_quantum.py` | Prediction script |
| `test_minimal_quantum.py` | Smoke tests |
