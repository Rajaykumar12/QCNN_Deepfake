"""
QViT Deepfake Detection — CNN Baseline Classifier
==================================================
Convolutional Neural Network that classifies directly from
preprocessed dual-channel images (128×128×2):
    Channel 0 — Grayscale [0, 1]
    Channel 1 — Log-DCT   [0, 1]

Architecture:
    • 4 Conv blocks (32 → 64 → 128 → 256 filters, BN + ReLU)
    • GlobalAveragePooling
    • Dense head → sigmoid
    Total: ~1.5M parameters
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from config import IMG_SIZE, HEAD_DROPOUT


def create_cnn_classifier(
    input_shape=None,
    head_dropout=HEAD_DROPOUT,
):
    """
    Build the CNN baseline classifier (Keras Functional API).

    Input:  (batch, 128, 128, 2)  preprocessed dual-channel images
    Output: (batch, 1)            probability of being fake
    """
    if input_shape is None:
        input_shape = (IMG_SIZE, IMG_SIZE, 2)

    inputs = tf.keras.layers.Input(shape=input_shape, name="preprocessed_input")

    # ── Conv Block 1 ─────────────────────────────
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool1")(x)          # → 64×64×32

    # ── Conv Block 2 ─────────────────────────────
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool2")(x)          # → 32×32×64

    # ── Conv Block 3 ─────────────────────────────
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool3")(x)          # → 16×16×128

    # ── Conv Block 4 ─────────────────────────────
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.Activation("relu", name="relu4")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool4")(x)          # → 8×8×256

    # ── Global pooling + head ─────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)     # → 256
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(head_dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_classifier")
    return model


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = create_cnn_classifier()
    model.summary()

    import numpy as np
    dummy = np.random.rand(2, IMG_SIZE, IMG_SIZE, 2).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    print(f"\nDummy input:  {dummy.shape}")
    print(f"Model output: {out.shape}  values: {out.flatten()}")
