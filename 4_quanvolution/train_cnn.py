"""
CNN Training on Quantum Features
==================================
Loads the .npy feature maps produced by quantum_preprocess.py and trains
a purely classical CNN to classify real vs fake images.

Architecture: Conv32 → BN → Pool → Conv64 → BN → Pool → Conv128 → BN → GAP
              → Dense64 → Dense1
~102k params. This configuration gave 68.4% val accuracy in prior experiments.

Usage:
    python train_cnn.py --features_dir ./quantum_features
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

OUTPUT_DIR = "./cnn_results"


# ── data ────────────────────────────────────────────────────────────────────

def load_features(features_dir):
    data = {}
    print("Loading features …")
    for split in ["train", "val", "test"]:
        X_path = os.path.join(features_dir, f"{split}_X.npy")
        y_path = os.path.join(features_dir, f"{split}_y.npy")
        if not os.path.isfile(X_path):
            raise FileNotFoundError(f"Not found: {X_path}\nRun quantum_preprocess.py first.")
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)
        pos = int(y.sum()); neg = len(y) - pos
        print(f"  {split}: X={X.shape}  y={y.shape}  (real={pos}, fake={neg})")
        data[split] = (X, y)
    return data


def make_dataset(X, y, batch_size, shuffle=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── model ────────────────────────────────────────────────────────────────────

def build_cnn(feature_shape):
    """
    102k-param CNN on quantum feature maps.

    Deliberately uses BatchNorm — the feature distribution from the quantum
    layer (shifted PauliZ values in [0,1]) is well-behaved enough for BN.
    This is the config that achieved 68.4% val accuracy.
    """
    reg = tf.keras.regularizers.l2(1e-4)
    inp = layers.Input(shape=feature_shape, name="input")

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x   = layers.Dropout(0.5, name="drop1")(x)
    x   = layers.Dense(64, activation="gelu", kernel_regularizer=reg, name="fc1")(x)
    x   = layers.Dropout(0.4, name="drop2")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inp, out, name="quantum_feature_cnn")


# ── training ─────────────────────────────────────────────────────────────────

def train(features_dir, output_dir,
          batch_size=32, epochs=100, learning_rate=1e-3):

    os.makedirs(output_dir, exist_ok=True)
    data = load_features(features_dir)

    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]

    feature_shape = X_train.shape[1:]
    print(f"\nFeature shape: {feature_shape}")

    train_ds = make_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   batch_size)
    test_ds  = make_dataset(X_test,  y_test,  batch_size)

    model = build_cnn(feature_shape)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(output_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=15,
                      restore_best_weights=True, verbose=1, mode="max"),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1, mode="max"),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1, mode="max"),
    ]

    print(f"\nTraining CNN on quantum features ({epochs} epochs) …")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=1,
    )

    # ── evaluation ──────────────────────────────────────────────────────────
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\n{'─'*50}")
    print(f"  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'─'*50}")

    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test loss    : {loss:.4f}\nTest accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")

    # ── training curves ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(history.history["loss"],         label="train")
    ax2.plot(history.history["val_loss"],     label="val")
    ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    plt.close()

    print(f"\n✅  Results saved to: {output_dir}")
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train CNN on quantum features")
    p.add_argument("--features_dir",   default="./quantum_features")
    p.add_argument("--output_dir",     default=OUTPUT_DIR)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--learning_rate",  type=float, default=1e-3)
    args = p.parse_args()

    if not os.path.isdir(args.features_dir):
        print(f"Features dir not found: {args.features_dir}")
        print("Run quantum_preprocess.py first."); return

    train(args.features_dir, args.output_dir,
          args.batch_size, args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
