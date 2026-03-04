"""
Hybrid Quantum-Classical CNN Training
======================================
Trains the dual-branch hybrid model defined in hybrid_quantum_cnn.py.

Usage:
    python train_hybrid_quantum.py --dataset_dir archive/Dataset

Dataset layout expected:
    archive/Dataset/
        Train/Real/   Train/Fake/
        Validation/Real/  Validation/Fake/
        Test/Real/    Test/Fake/
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

from hybrid_quantum_cnn import create_hybrid_quantum_cnn
from minimal_quantum_cnn import MinimalQuantumLayer   # needed for model.save/load
from config import (
    CLASSES, RANDOM_SEED, IMG_SIZE,
    MAX_SAMPLES_PER_CLASS, BATCH_SIZE, EPOCHS,
    EARLY_STOPPING_PATIENCE, HYBRID_OUTPUT_DIR,
)

SPLIT_DIRS = {"train": "Train", "val": "Validation", "test": "Test"}


# ── data loading ─────────────────────────────────────────────────────────────

def load_images(split_dir, img_size=IMG_SIZE, max_per_class=MAX_SAMPLES_PER_CLASS):
    """Load grayscale images from split_dir/ClassName/ into numpy arrays."""
    images, labels = [], []
    print(f"  Loading from: {split_dir}")
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"    ⚠  {cls_dir} not found — skipping"); continue
        count = 0
        for root, _, files in os.walk(cls_dir):
            for fname in files:
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                try:
                    img = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img.astype(np.float32) / 255.0)
                    labels.append(label_idx)
                    count += 1
                    if count >= max_per_class:
                        break
                except Exception:
                    continue
            if count >= max_per_class:
                break
        print(f"    {cls}: {count}")

    images = np.asarray(images)[..., np.newaxis]   # (N, H, W, 1)
    labels = np.asarray(labels, dtype=np.int32)
    return images, labels


# ── augmentation ─────────────────────────────────────────────────────────────

def _augment(image, label):
    """
    Minimal augmentation — only horizontal flip.

    Deepfake artifacts live in high-frequency texture (blending seams,
    GAN fingerprints).  Brightness/contrast/rotation corrupt these cues, so
    we keep only the one transform that is semantically plausible for faces.
    """
    image = tf.image.random_flip_left_right(image)
    return image, label


def make_dataset(images, labels, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (images.astype(np.float32), labels.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=RANDOM_SEED)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── training ─────────────────────────────────────────────────────────────────

def train(dataset_dir,
          output_dir=HYBRID_OUTPUT_DIR,
          img_size=IMG_SIZE,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          max_per_class=MAX_SAMPLES_PER_CLASS,
          learning_rate=5e-5):

    os.makedirs(output_dir, exist_ok=True)

    val_cap  = min(max_per_class, 2000)
    test_cap = min(max_per_class, 1000)

    print("Train split:");  X_train, y_train = load_images(
        os.path.join(dataset_dir, SPLIT_DIRS["train"]), img_size, max_per_class)
    print("Validation split:"); X_val, y_val = load_images(
        os.path.join(dataset_dir, SPLIT_DIRS["val"]),   img_size, val_cap)
    print("Test split:");    X_test, y_test = load_images(
        os.path.join(dataset_dir, SPLIT_DIRS["test"]),  img_size, test_cap)

    print(f"\nSizes → train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")
    if len(X_train) == 0:
        raise ValueError("No training images found — check dataset path.")

    train_ds = make_dataset(X_train, y_train, batch_size, shuffle=True,  augment=True)
    val_ds   = make_dataset(X_val,   y_val,   batch_size)
    test_ds  = make_dataset(X_test,  y_test,  batch_size)

    print("\nBuilding hybrid model …")
    model = create_hybrid_quantum_cnn((img_size, img_size, 1))
    model.summary()

    # Lower default LR than the sequential model.
    # Classical branch converges faster than quantum; starting at 5e-5
    # prevents the classical branch from racing ahead and making quantum
    # gradients irrelevant before quantum weights have had a chance to orient.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,          # tame quantum gradient spikes
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1, mode="max"),
        ModelCheckpoint(os.path.join(output_dir, "best_model.keras"),
                        monitor="val_accuracy", save_best_only=True,
                        verbose=1, mode="max"),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1, mode="max"),
    ]

    print(f"\nTraining for up to {epochs} epochs …")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=1,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest loss {test_loss:.4f}  |  Test accuracy {test_acc:.4f}")

    model.save(os.path.join(output_dir, "final_model.keras"))
    np.save(os.path.join(output_dir, "history.npy"), history.history)
    _save_plots(history, output_dir)

    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"test_loss: {test_loss:.6f}\ntest_accuracy: {test_acc:.6f}\n")

    print(f"Artefacts saved to {output_dir}")
    return model, history


# ── plotting ─────────────────────────────────────────────────────────────────

def _save_plots(history, output_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history["accuracy"],    label="train")
    a1.plot(history.history["val_accuracy"], label="val")
    a1.set(title="Accuracy", xlabel="epoch", ylabel="accuracy"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(history.history["loss"],    label="train")
    a2.plot(history.history["val_loss"], label="val")
    a2.set(title="Loss",     xlabel="epoch", ylabel="loss");     a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train Hybrid Quantum-Classical CNN")
    p.add_argument("--dataset_dir",   default="archive/Dataset")
    p.add_argument("--output_dir",    default=HYBRID_OUTPUT_DIR)
    p.add_argument("--img_size",      type=int,   default=IMG_SIZE)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--max_samples",   type=int,   default=MAX_SAMPLES_PER_CLASS)
    p.add_argument("--learning_rate", type=float, default=5e-5,
                   help="Initial LR (default 5e-5; lower than sequential model "
                        "to let quantum branch orient before classical takes over)")
    args = p.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset dir not found: {args.dataset_dir}"); return
    for key, sub in SPLIT_DIRS.items():
        d = os.path.join(args.dataset_dir, sub)
        if not os.path.isdir(d):
            print(f"Missing split dir: {d}"); return

    train(args.dataset_dir, args.output_dir, args.img_size,
          args.batch_size, args.epochs, args.max_samples, args.learning_rate)


if __name__ == "__main__":
    main()
