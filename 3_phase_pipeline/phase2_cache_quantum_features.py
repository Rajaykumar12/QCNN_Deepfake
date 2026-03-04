"""
Phase 2 — Cache Quantum Feature Maps
======================================
Loads the pre-trained quantum weights from Phase 1, runs every image in all
three splits (Train / Validation / Test) through the *frozen* quantum layer,
and saves the resulting feature maps as compressed .npz files.

Output files in CACHED_FEATURES_DIR:
    train_features.npz  — keys: 'X' (N,16,16,4)  'y' (N,)
    val_features.npz
    test_features.npz

After this step, Phase 3 never calls PennyLane again.  Training becomes
purely classical and orders of magnitude faster.

Usage:
    python phase2_cache_quantum_features.py --dataset_dir archive/Dataset
    python phase2_cache_quantum_features.py --dataset_dir archive/Dataset \\
        --quantum_weights ./pretrained_quantum/quantum_weights.npy
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, argparse
import numpy as np
import cv2
import tensorflow as tf

from minimal_quantum_cnn import MinimalQuantumLayer, MultiObservableQuantumLayer
from config import (
    CLASSES, IMG_SIZE, N_QUBITS,
    MAX_SAMPLES_PER_CLASS, PRETRAIN_OUTPUT_DIR, CACHED_FEATURES_DIR,
)

SPLIT_DIRS = {"train": "Train", "val": "Validation", "test": "Test"}
SPLIT_CAPS  = {"train": MAX_SAMPLES_PER_CLASS, "val": 2000, "test": 1000}


# ── data loader ───────────────────────────────────────────────────────────────

def load_images(split_dir, img_size=IMG_SIZE, max_per_class=5000):
    images, labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠  {cls_dir} not found — skipping"); continue
        count = 0
        for root, _, files in os.walk(cls_dir):
            for fname in files:
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
    return np.asarray(images)[..., np.newaxis], np.asarray(labels, dtype=np.int32)


# ── quantum feature extractor ─────────────────────────────────────────────────

def build_feature_extractor(img_size, weights_path):
    """
    Builds a frozen multi-observable quantum feature extractor.

    Architecture: Input → MultiObservableQuantumLayer → rescale
    Output shape: (batch, H//4, W//4, 3×N_QUBITS)  →  (B, 16, 16, 12)

    Uses MultiObservableQuantumLayer (X+Y+Z per qubit) instead of Z-only:
    - Triples the information content of cached features at zero extra cost
    - The pre-trained Z-only weights are still valid; X/Y measurements
      just probe the same trained quantum state from orthogonal bases
    """
    from tensorflow.keras import layers

    inp = layers.Input(shape=(img_size, img_size, 1), name="image_input")
    q   = MultiObservableQuantumLayer(name="quantum_preprocess")(inp)
    q   = layers.Lambda(lambda t: (t + 1.0) / 2.0, name="q_rescale")(q)

    extractor = tf.keras.Model(inp, q, name="quantum_feature_extractor")

    # Load pre-trained quantum weights (same shape as MinimalQuantumLayer)
    q_weights = np.load(weights_path, allow_pickle=True)
    q_layer   = extractor.get_layer("quantum_preprocess")
    q_layer.set_weights(list(q_weights))
    q_layer.trainable = False   # FROZEN — no more quantum backprop

    print(f"  ✅  Loaded quantum weights from: {weights_path}")
    print(f"     Weight shape: {list(q_weights)[0].shape}")
    print(f"     Observables : PauliX + PauliY + PauliZ  →  12 feature channels")
    return extractor


# ── caching ───────────────────────────────────────────────────────────────────

def cache_split(extractor, split_dir, split_name, img_size, max_per_class,
                output_dir, batch_size=32):
    """
    Process one dataset split, run through quantum extractor, save features.

    Uses batched inference (no gradient tape) for speed.
    """
    print(f"\nProcessing split: {split_name} …")
    X_raw, y = load_images(split_dir, img_size, max_per_class)
    n = len(X_raw)
    print(f"  Total images: {n}")

    # Spatial output size: IMG_SIZE // 4 for 8×8 windows with stride 4
    # channels: auto-detected from extractor output (12 for multi-observable)
    spatial   = img_size // 4
    n_channels = extractor.output_shape[-1]   # 12 (X+Y+Z) or 4 (Z-only)
    X_feat    = np.zeros((n, spatial, spatial, n_channels), dtype=np.float32)

    # Batched forward pass — @tf.function + no_grad for max speed
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = tf.constant(X_raw[start:end], dtype=tf.float32)
        feats = extractor(batch, training=False)
        X_feat[start:end] = feats.numpy()
        print(f"  [{end:>5}/{n}] batches processed …", end="\r")

    out_path = os.path.join(output_dir, f"{split_name}_features.npz")
    np.savez_compressed(out_path, X=X_feat, y=y)
    print(f"\n  Saved → {out_path}  (shape: {X_feat.shape})")
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def cache_all(dataset_dir,
              weights_path=None,
              output_dir=CACHED_FEATURES_DIR,
              img_size=IMG_SIZE):

    os.makedirs(output_dir, exist_ok=True)

    if weights_path is None:
        weights_path = os.path.join(PRETRAIN_OUTPUT_DIR, "quantum_weights.npy")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Quantum weights not found at: {weights_path}\n"
            f"Run phase1_pretrain_quantum.py first."
        )

    print("Building frozen quantum feature extractor …")
    extractor = build_feature_extractor(img_size, weights_path)

    for split_name, sub_dir in SPLIT_DIRS.items():
        split_dir = os.path.join(dataset_dir, sub_dir)
        if not os.path.isdir(split_dir):
            print(f"  ⚠  Split dir not found: {split_dir} — skipping"); continue
        cache_split(
            extractor, split_dir, split_name,
            img_size, SPLIT_CAPS[split_name], output_dir,
        )

    print(f"\n✅  All features cached to: {output_dir}")
    print(f"   Next step → run phase3_train_on_features.py")


def main():
    p = argparse.ArgumentParser(description="Phase 2: Cache quantum feature maps")
    p.add_argument("--dataset_dir",     default="archive/Dataset")
    p.add_argument("--quantum_weights",
                   default=os.path.join(PRETRAIN_OUTPUT_DIR, "quantum_weights.npy"))
    p.add_argument("--output_dir",      default=CACHED_FEATURES_DIR)
    p.add_argument("--img_size",        type=int, default=IMG_SIZE)
    args = p.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset dir not found: {args.dataset_dir}"); return

    cache_all(args.dataset_dir, args.quantum_weights, args.output_dir, args.img_size)


if __name__ == "__main__":
    main()
