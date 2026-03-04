"""
Phase 1 — Pre-train Quantum Weights in Isolation
=================================================
Trains ONLY the quantum circuit weights using a tiny linear classifier:

    Image (64×64×1)
    → MinimalQuantumLayer    (learnable quantum weights)
    → GlobalAveragePooling2D (collapses spatial dims → 4 values)
    → Dense(1, sigmoid)      (binary classifier)

Why this matters
----------------
When quantum weights are trained end-to-end with a large CNN, the classical
gradients dominate and quantum weights never properly orient.  By training the
quantum layer *alone* first, we let the 24 quantum parameters converge to a
configuration that genuinely discriminates real vs fake faces before any CNN
is introduced.  The saved weights are then used to initialise Phase 2 & 3.

Usage:
    python phase1_pretrain_quantum.py --dataset_dir archive/Dataset
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from minimal_quantum_cnn import MinimalQuantumLayer
from config import (
    CLASSES, RANDOM_SEED, IMG_SIZE, N_QUBITS,
    MAX_SAMPLES_PER_CLASS, PRETRAIN_OUTPUT_DIR,
)

SPLIT_DIRS = {"train": "Train", "val": "Validation"}

# ── data ─────────────────────────────────────────────────────────────────────

def load_images(split_dir, img_size=IMG_SIZE, max_per_class=2000):
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


def make_dataset(images, labels, batch_size=8, shuffle=False):
    """Small batch size — quantum backprop is expensive, keep memory low."""
    ds = tf.data.Dataset.from_tensor_slices(
        (images.astype(np.float32), labels.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(len(images), seed=RANDOM_SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ── model ─────────────────────────────────────────────────────────────────────

def build_quantum_only_model(img_size=IMG_SIZE):
    """
    Minimal model: only the quantum layer + a linear head.

    The GAP+Dense head is intentionally simple — it collapses spatial dims to
    N_QUBITS scalars and lets a single linear layer classify.  This simplicity
    is a feature: it maximises the gradient signal flowing into the 24 quantum
    parameters without competing classical layers consuming gradient budget.

    Conv-head variants were tested and consistently performed worse (stuck at
    50% during quantum-only warm-up because a frozen Conv produces near-zero
    gradient variance for the quantum weights).
    """
    inp = layers.Input(shape=(img_size, img_size, 1), name="image_input")
    x   = MinimalQuantumLayer(name="quantum_preprocess")(inp)
    x   = layers.Lambda(lambda t: (t + 1.0) / 2.0, name="q_rescale")(x)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)   # (B, N_QUBITS)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inp, out, name="quantum_only")

# ── training ─────────────────────────────────────────────────────────────────

def pretrain(dataset_dir,
             output_dir=PRETRAIN_OUTPUT_DIR,
             img_size=IMG_SIZE,
             max_per_class=2000,
             epochs=40):

    os.makedirs(output_dir, exist_ok=True)

    print("Train split:")
    X_train, y_train = load_images(
        os.path.join(dataset_dir, SPLIT_DIRS["train"]), img_size, max_per_class)
    print("Validation split:")
    X_val, y_val = load_images(
        os.path.join(dataset_dir, SPLIT_DIRS["val"]),   img_size, 500)

    print(f"\nSizes → train {len(X_train)}  val {len(X_val)}")
    if len(X_train) == 0:
        raise ValueError("No training images found.")

    BATCH = 8
    train_ds = make_dataset(X_train, y_train, BATCH, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   BATCH)

    model = build_quantum_only_model(img_size)
    model.summary()

    # Higher LR: only 24+1 params, no competing classical gradients.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(output_dir, "best_quantum_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1, mode="max"),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1, mode="max"),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1, mode="max"),
    ]

    print(f"\nPre-training quantum weights for up to {epochs} epochs …")
    model.fit(train_ds, validation_data=val_ds,
              epochs=epochs, callbacks=callbacks, verbose=1)


    # ── Extract and save ONLY the quantum weights ──────────────────────────
    q_layer  = model.get_layer("quantum_preprocess")
    q_weights = q_layer.get_weights()   # list of numpy arrays
    weights_path = os.path.join(output_dir, "quantum_weights.npy")
    np.save(weights_path, np.array(q_weights, dtype=object), allow_pickle=True)
    print(f"\n✅  Quantum weights saved to: {weights_path}")
    print(f"   Weight shape: {q_weights[0].shape}  "
          f"(N_LAYERS={q_weights[0].shape[0]}, N_QUBITS={q_weights[0].shape[1]}, 3 angles)")
    print(f"\nNext step → run phase2_cache_quantum_features.py")
    return model, q_weights


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 1: Pre-train quantum weights")
    p.add_argument("--dataset_dir",   default="archive/Dataset")
    p.add_argument("--output_dir",    default=PRETRAIN_OUTPUT_DIR)
    p.add_argument("--img_size",      type=int, default=IMG_SIZE)
    p.add_argument("--max_samples",   type=int, default=2000,
                   help="Max images per class for quantum pre-training (default 2000)")
    p.add_argument("--epochs",        type=int, default=40)
    args = p.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset dir not found: {args.dataset_dir}"); return

    pretrain(args.dataset_dir, args.output_dir, args.img_size,
             args.max_samples, args.epochs)


if __name__ == "__main__":
    main()
