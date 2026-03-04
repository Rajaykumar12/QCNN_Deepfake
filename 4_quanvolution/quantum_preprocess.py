"""
Quantum Preprocessing (Quanvolution)
======================================
Adapts the quanvolution approach from:
    "Quanvolutional Neural Networks" (Henderson et al., 2020)

For each image:
  1. Slide a 4×4 window across the 64×64 image with stride 4
     → 16×16 spatial positions
  2. At each position, extract 4 corner pixels as qubit inputs
  3. Pass them through a random quantum circuit (RandomLayers template)
  4. Collect PauliZ expectation values → 4 features per position
  → Output: (16, 16, 4) quantum feature map per image

Uses lightning.qubit (C++ backend) for faster simulation.
Falls back to default.qubit if lightning is not installed.

Output .npy files:
    quantum_features/<split>_X.npy   shape (N, 16, 16, 4)  float32
    quantum_features/<split>_y.npy   shape (N,)             int32

Usage:
    python quantum_preprocess.py --dataset_dir archive/Dataset
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, argparse
import numpy as np
import cv2
import pennylane as qml
import tensorflow as tf
from config import CLASSES, IMG_SIZE, N_QUBITS

OUTPUT_DIR  = "./quantum_features"
SPLIT_DIRS  = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_CAPS  = {"train": 5000,    "val": 2000,         "test": 1000}
PATCH_SIZE  = 4      # 4×4 pixel window per quantum circuit call
STRIDE      = 4      # non-overlapping → output is 64//4 = 16 × 16
N_LAYERS    = 3      # RandomLayers depth (more layers → richer mixing)
BATCH_SIZE  = 32


# ── quantum circuit ───────────────────────────────────────────────────────────

def build_circuit(n_qubits, n_layers, seed=42):
    """
    Build a random quantum circuit using PennyLane's RandomLayers template.
    The circuit parameters are fixed at init time (random but reproducible).

    RandomLayers applies random RY/RZ rotations and CNOT entangling gates
    — a clean, parameterised alternative to a hand-coded Rot+CNOT stack.
    """
    np.random.seed(seed)
    rand_params = np.random.uniform(
        high=2 * np.pi, size=(n_layers, n_qubits)
    ).astype(np.float64)   # shape (n_layers, n_qubits)

    # Try lightning.qubit (C++ backend, ~10× faster on CPU).
    # Fall back to default.qubit if not installed.
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print(f"✅  Using lightning.qubit ({n_qubits} wires)")
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        print(f"ℹ️   lightning.qubit not available — using default.qubit")
        print("    Install with: pip install pennylane-lightning")

    @qml.qnode(dev)
    def circuit(phi):
        """
        phi : (n_qubits,) pixel values in [0, 1]

        Phase encoding: RY(pi * phi) maps [0,1] → [0, pi]
        — the same encoding used in the reference quanvolution paper.
        """
        # Encode pixel values as qubit rotations
        for j in range(n_qubits):
            qml.RY(np.pi * phi[j], wires=j)
        # Apply random entangling layers
        qml.RandomLayers(weights=rand_params, wires=list(range(n_qubits)),
                         seed=seed)
        # Measure all qubits in the Z basis
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

    return circuit


# ── quanvolution ──────────────────────────────────────────────────────────────

def quanv(image, circuit, n_qubits, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Apply quantum circuit as a sliding-window filter (quanvolution).

    image  : (H, W) numpy float32, values in [0, 1]
    returns: (H//stride, W//stride, n_qubits) float32

    For each stride×stride window we feed corner pixel values into the
    quantum circuit and collect PauliZ expectation values as features.
    """
    H, W    = image.shape
    out_H   = H // stride
    out_W   = W // stride
    out     = np.zeros((out_H, out_W, n_qubits), dtype=np.float32)

    for row in range(out_H):
        for col in range(out_W):
            r = row * stride
            c = col * stride
            # 4 corner pixels of the patch — same pattern as original circuit
            pixels = np.array([
                image[r,              c            ],   # top-left
                image[r,              min(c+patch_size-1, W-1)],  # top-right
                image[min(r+patch_size-1, H-1), c ],   # bottom-left
                image[min(r+patch_size-1, H-1), min(c+patch_size-1, W-1)],
            ], dtype=np.float64)[:n_qubits]             # clip to n_qubits
            q_out = circuit(pixels)
            for k in range(n_qubits):
                out[row, col, k] = float(q_out[k])

    # Shift from [-1,1] → [0,1] (same normalisation as original pipeline)
    return (out + 1.0) / 2.0


# ── data loading ──────────────────────────────────────────────────────────────

def load_images(split_dir, img_size, max_per_class):
    images, labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠  {cls_dir} not found — skipping"); continue
        count = 0
        for root, _, files in os.walk(cls_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                try:
                    img = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img.astype(np.float32) / 255.0)
                    labels.append(label_idx); count += 1
                    if count >= max_per_class: break
                except Exception: continue
            if count >= max_per_class: break
        print(f"    {cls}: {count}")
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)


# ── caching ───────────────────────────────────────────────────────────────────

def cache_split(circuit, split_dir, split_name, img_size,
                max_per_class, output_dir, n_qubits):
    print(f"\n── {split_name} ──")
    X_raw, y = load_images(split_dir, img_size, max_per_class)
    n = len(X_raw)
    if n == 0:
        print("  No images found, skipping."); return

    out_size  = img_size // STRIDE
    X_feat    = np.zeros((n, out_size, out_size, n_qubits), dtype=np.float32)

    for i, img in enumerate(X_raw):
        X_feat[i] = quanv(img, circuit, n_qubits)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{i+1:>5}/{n}]", end="\r")

    X_path = os.path.join(output_dir, f"{split_name}_X.npy")
    y_path = os.path.join(output_dir, f"{split_name}_y.npy")
    np.save(X_path, X_feat)
    np.save(y_path, y)
    print(f"\n  ✅  {split_name}: {X_feat.shape}  pos={int(y.sum())} neg={n-int(y.sum())}")


def run(dataset_dir, output_dir, img_size, n_qubits, n_layers):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nBuilding quantum circuit ({n_qubits} qubits, {n_layers} RandomLayers) …")
    circuit = build_circuit(n_qubits, n_layers)

    spatial = img_size // STRIDE
    print(f"Image {img_size}×{img_size} → features {spatial}×{spatial}×{n_qubits}")

    for split, sub_dir in SPLIT_DIRS.items():
        split_dir = os.path.join(dataset_dir, sub_dir)
        if not os.path.isdir(split_dir):
            print(f"  ⚠  {split_dir} not found — skipping"); continue
        cache_split(circuit, split_dir, split, img_size,
                    SPLIT_CAPS[split], output_dir, n_qubits)

    print(f"\n✅  All features saved to: {output_dir}")
    print(f"   Next: python train_cnn.py --features_dir {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Quantum preprocessing (quanvolution)")
    p.add_argument("--dataset_dir", default="archive/Dataset")
    p.add_argument("--output_dir",  default=OUTPUT_DIR)
    p.add_argument("--img_size",    type=int, default=IMG_SIZE)
    p.add_argument("--n_qubits",    type=int, default=N_QUBITS,
                   help="Number of qubits (= output channels)")
    p.add_argument("--n_layers",    type=int, default=N_LAYERS,
                   help="Depth of RandomLayers")
    args = p.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset dir not found: {args.dataset_dir}"); return

    run(args.dataset_dir, args.output_dir,
        args.img_size, args.n_qubits, args.n_layers)


if __name__ == "__main__":
    main()
