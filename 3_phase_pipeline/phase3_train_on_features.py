"""
Phase 3 — Train Classical CNN on Cached Quantum Features
=========================================================
Loads the pre-computed quantum feature maps from Phase 2 (.npz files) and
trains a classical convolutional network on top of them.

Because no quantum simulation happens here, each epoch takes seconds instead
of minutes.  The entire training run typically finishes in < 5 minutes.

Input features:  (N, 16, 16, 4)  — quantum expectation values ∈ [0, 1]
Output:          binary label (Real=0, Fake=1)

Network:
    Conv(32, 3) → BN → ReLU → Pool(2)
    Conv(64, 3) → BN → ReLU → Pool(2)
    Conv(128,3) → BN → ReLU → GAP
    Dropout(0.5) → Dense(64, gelu) → Dropout(0.4) → Dense(1, sigmoid)

Usage:
    python phase3_train_on_features.py
    python phase3_train_on_features.py --features_dir ./cached_quantum_features
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

from config import RANDOM_SEED, CACHED_FEATURES_DIR, PHASE3_OUTPUT_DIR, N_QUBITS

# ── data ─────────────────────────────────────────────────────────────────────

def load_features(features_dir, split_name):
    """Load pre-computed quantum feature maps for one split."""
    path = os.path.join(features_dir, f"{split_name}_features.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Features not found: {path}\n"
            f"Run phase2_cache_quantum_features.py first."
        )
    data = np.load(path)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    print(f"  {split_name}: X={X.shape}  y={y.shape}  "
          f"  (pos={int(y.sum())}, neg={int((1-y).sum())})")
    return X, y


def _augment(image, label):
    """Tiny augmentation on quantum features — only horizontal flip is valid."""
    image = tf.image.random_flip_left_right(image)
    return image, label


def make_dataset(X, y, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=RANDOM_SEED)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── model ─────────────────────────────────────────────────────────────────────

def build_feature_cnn(feature_shape):
    """
    Medium-capacity CNN with LayerNormalization for quantum feature maps.

    Input  : (H//4, W//4, N_QUBITS)  e.g. (16, 16, 4)
    Output : (1,) sigmoid probability
    Params : ~30k

    Key design decision — LayerNorm over BatchNorm:
      BatchNorm maintains running statistics (mean/var) computed on training
      batches.  Quantum feature maps have an unusual fixed distribution (PauliZ
      expectations rescaled to [0,1]) that is consistent across samples but
      whose batch statistics fluctuate significantly with the small 4-channel
      input.  This mismatch between train running stats and inference stats
      was the root cause of val_accuracy collapsing to 50% every few epochs.
      LayerNorm normalises per-sample at both train AND test time — no running
      stats, no mismatch, no spikes.

    Size is deliberately between the 6.5k (underfit) and 102k (overfit) runs:
      Conv(32) → LN → Pool → Conv(64) → LN → GAP → Dense(32) → Dense(1)
    """
    reg = tf.keras.regularizers.l2(2e-4)
    inp = layers.Input(shape=feature_shape, name="quantum_features")

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv1")(inp)
    x = layers.LayerNormalization(name="ln1")(x)   # LayerNorm: no running stats
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)    # (8, 8, 32)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, name="conv2")(x)
    x = layers.LayerNormalization(name="ln2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # (64,)

    x   = layers.Dropout(0.5, name="drop1")(x)
    x   = layers.Dense(32, activation="gelu",
                        kernel_regularizer=reg, name="fc1")(x)
    x   = layers.Dropout(0.4, name="drop2")(x)
    out = layers.Dense(1, activation="sigmoid",
                       kernel_regularizer=reg, name="output")(x)

    return tf.keras.Model(inp, out, name="quantum_feature_cnn")


# ── training ─────────────────────────────────────────────────────────────────

def train(features_dir=CACHED_FEATURES_DIR,
          output_dir=PHASE3_OUTPUT_DIR,
          batch_size=32,
          epochs=100,
          learning_rate=1e-3):

    os.makedirs(output_dir, exist_ok=True)

    print("Loading cached quantum features …")
    X_train, y_train = load_features(features_dir, "train")
    X_val,   y_val   = load_features(features_dir, "val")
    X_test,  y_test  = load_features(features_dir, "test")

    feature_shape = X_train.shape[1:]   # e.g. (16, 16, 4)
    print(f"\nFeature map shape: {feature_shape}")

    train_ds = make_dataset(X_train, y_train, batch_size, shuffle=True, augment=True)
    val_ds   = make_dataset(X_val,   y_val,   batch_size)
    test_ds  = make_dataset(X_test,  y_test,  batch_size)

    model = build_feature_cnn(feature_shape)
    model.summary()

    # Lower LR (3e-4 default) to prevent the calibration crashes seen at 1e-3
    # (val_loss spiking to 2.1+, val_accuracy dropping to 50% mid-training).
    # Add clipnorm=1.0 as a safety net against any remaining loss spikes.
    # Label smoothing 0.1 provides stronger regularisation for the smaller model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=15,
                      restore_best_weights=True, verbose=1, mode="max"),
        ModelCheckpoint(os.path.join(output_dir, "best_model.keras"),
                        monitor="val_accuracy", save_best_only=True,
                        verbose=1, mode="max"),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=7, min_lr=1e-7, verbose=1, mode="max"),
    ]

    print(f"\nTraining for up to {epochs} epochs (pure classical — fast!) …")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=1,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n{'─'*50}")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"{'─'*50}")

    model.save(os.path.join(output_dir, "final_model.keras"))
    np.save(os.path.join(output_dir, "history.npy"), history.history)
    _save_plots(history, output_dir)

    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"test_loss: {test_loss:.6f}\ntest_accuracy: {test_acc:.6f}\n")

    print(f"\n✅  Artefacts saved to: {output_dir}")
    return model, history, test_acc


# ── plotting ─────────────────────────────────────────────────────────────────

def _save_plots(history, output_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history["accuracy"],     label="train")
    a1.plot(history.history["val_accuracy"], label="val")
    a1.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    a1.legend(); a1.grid(alpha=.3)
    a2.plot(history.history["loss"],     label="train")
    a2.plot(history.history["val_loss"], label="val")
    a2.set(title="Loss", xlabel="epoch", ylabel="loss")
    a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Phase 3: Train classical CNN on cached quantum features")
    p.add_argument("--features_dir",  default=CACHED_FEATURES_DIR)
    p.add_argument("--output_dir",    default=PHASE3_OUTPUT_DIR)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--learning_rate", type=float, default=3e-4,
                   help="Initial LR. Default 3e-4 (lower than before to prevent "
                        "calibration crashes that dropped val_accuracy to 50%)")
    args = p.parse_args()

    train(args.features_dir, args.output_dir,
          args.batch_size, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
