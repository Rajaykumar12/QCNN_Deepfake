"""
QViT Deepfake Detection — CNN Baseline Training
================================================
Trains the CNN classifier directly on preprocessed dual-channel
images (128×128×2). Reads from PREPROCESSED_DIR — no quantum
convolution step needed.

Usage:
    python train_cnn.py
    python train_cnn.py --input_dir /path/to/preprocessed --output_dir /path/to/output
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import math
import argparse
import numpy as np
import tensorflow as tf

from config import (
    PREPROCESSED_DIR, BATCH_SIZE, EPOCHS,
    WEIGHT_DECAY, WARMUP_EPOCHS, EARLY_STOPPING_PATIENCE,
)
from cnn_model import create_cnn_classifier


# ──────────────────────────────────────────────
# Learning Rate Schedule  (same as ViT trainer)
# ──────────────────────────────────────────────

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, initial_lr, warmup_steps, decay_steps, min_lr=1e-6):
        super().__init__()
        self.initial_lr  = float(initial_lr)
        self.warmup_steps = float(warmup_steps)
        self.decay_steps  = float(decay_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_pct = tf.minimum(step / tf.maximum(self.warmup_steps, 1.0), 1.0)
        warmup_lr  = self.initial_lr * warmup_pct

        decay_step = tf.maximum(step - self.warmup_steps, 0.0)
        decay_pct  = tf.minimum(decay_step / tf.maximum(self.decay_steps, 1.0), 1.0)
        cosine_lr  = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * decay_pct)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":   self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps":  self.decay_steps,
            "min_lr":       self.min_lr,
        }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default=None,
                        help="Path to preprocessed numpy arrays (overrides config)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save model + history")
    args, _ = parser.parse_known_args()

    # Azure ML env-var fallback, then argparse, then config default
    preprocessed_dir = (
        args.input_dir
        or os.environ.get("AZUREML_PREPROCESSED_DIR")
        or PREPROCESSED_DIR
    )
    output_dir = (
        args.output_dir
        or os.environ.get("AZUREML_MODEL_DIR")
        or "./content"
    )

    model_save_path   = os.path.join(output_dir, "cnn_model.keras")
    history_save_path = os.path.join(output_dir, "cnn_training_history.npy")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TRAINING — CNN Deepfake Classifier")
    print("=" * 60)

    # ── Load preprocessed images ─────────────────
    print(f"\n📂 Loading preprocessed images from: {preprocessed_dir}")
    train_images = np.load(os.path.join(preprocessed_dir, "train_images.npy"))
    val_images   = np.load(os.path.join(preprocessed_dir, "val_images.npy"))
    train_labels = np.load(os.path.join(preprocessed_dir, "train_labels.npy"))
    val_labels   = np.load(os.path.join(preprocessed_dir, "val_labels.npy"))

    train_images = train_images.astype(np.float32)
    val_images   = val_images.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    val_labels   = val_labels.astype(np.float32)

    print(f"  Train: {train_images.shape}, labels: {train_labels.shape}")
    print(f"  Val:   {val_images.shape},   labels: {val_labels.shape}")
    print(f"  Input shape: {train_images.shape[1:]}")

    # ── Class weights ────────────────────────────
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    class_weight = {
        int(cls): total / (len(unique) * count)
        for cls, count in zip(unique, counts)
    }
    print(f"  Class weights: {class_weight}")

    # ── Build model ──────────────────────────────
    print("\n🧠 Building CNN classifier...")
    model = create_cnn_classifier(input_shape=train_images.shape[1:])
    model.summary()

    # ── LR schedule ──────────────────────────────
    steps_per_epoch = math.ceil(len(train_images) / BATCH_SIZE)
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
    decay_steps     = (EPOCHS - WARMUP_EPOCHS) * steps_per_epoch

    lr_schedule = WarmupCosineSchedule(
        initial_lr=3e-4,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr=1e-6,
    )

    # ── Compile ──────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=WEIGHT_DECAY,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # ── Callbacks ────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Epochs:        {EPOCHS}  (early stop patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Steps/epoch:   {steps_per_epoch}")
    print(f"  Initial LR:    3e-4")
    print(f"  Weight decay:  {WEIGHT_DECAY}")

    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save history ─────────────────────────────
    history_dict = {
        key: [float(v) for v in vals]
        for key, vals in history.history.items()
    }
    np.save(history_save_path, history_dict)
    print(f"\n✓ Training history saved to: {history_save_path}")

    # ── Final evaluation ─────────────────────────
    val_loss, val_acc, val_auc = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\nFinal val accuracy: {val_acc:.4f}")
    print(f"Final val AUC:      {val_auc:.4f}")
    print(f"Final val loss:     {val_loss:.4f}")
    print(f"\n✓ Best model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
