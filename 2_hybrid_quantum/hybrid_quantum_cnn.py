"""
Hybrid Quantum-Classical CNN for Deepfake Detection
====================================================
Parallel dual-branch architecture:

  - Classical branch: standard CNN, provides stable gradients and strong
    spatial feature extraction.
  - Quantum branch:   MinimalQuantumLayer preprocessing + small CNN, adds
    quantum-specific correlations as complementary features.

Both branches receive the same input and their GAP outputs are concatenated
before a small MLP classifier.

Key motivation vs sequential (quantum → CNN):
  In the sequential design, quantum outputs are the ONLY input to the CNN.
  When quantum weights enter a barren plateau and produce near-zero gradients,
  the whole model loses signal. Here, the classical branch always provides
  strong gradients, stabilising training even in bad quantum update steps.
  The model can also learn to weight each branch: if quantum features are
  uninformative at some stage, the Dense layers simply down-weight them.

Architecture summary
--------------------
Input (B, 64, 64, 1)
├─ Classical branch
│   Conv2D(32)→BN→ReLU→Pool → Conv2D(64)→BN→ReLU→GAP  →  64-d
└─ Quantum branch
    QuantumLayer(16×16×4) → rescale → Conv2D(16)→BN→LReLU→Pool
                          → Conv2D(32)→BN→ReLU→GAP              →  32-d
Concat(96-d) → Dropout(0.5) → Dense(48, gelu) → Dropout(0.4) → Dense(1, sigmoid)

Trainable params (approx):
  Quantum weights  :   24   (N_LAYERS=2 × N_QUBITS=4 × 3)
  Quantum CNN body :  ~5 k
  Classical branch : ~55 k
  MLP head         :  ~5 k
  Total            : ~65 k
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers

from config import IMG_SIZE, N_QUBITS, N_LAYERS
# Re-use the quantum layer from the existing module.
from minimal_quantum_cnn import MinimalQuantumLayer


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_hybrid_quantum_cnn(input_shape=None):
    """
    Build and return the hybrid quantum-classical model.

    Parameters
    ----------
    input_shape : tuple, optional
        (H, W, C) of the input image.  Defaults to (IMG_SIZE, IMG_SIZE, 1).

    Returns
    -------
    tf.keras.Model
    """
    if input_shape is None:
        input_shape = (IMG_SIZE, IMG_SIZE, 1)

    reg_l2 = tf.keras.regularizers.l2(1e-4)
    inp = layers.Input(shape=input_shape, name="image_input")

    # ── Classical branch ──────────────────────────────────────────────────
    # Intentionally kept moderately sized so the quantum branch can still
    # contribute meaningful features rather than being drowned out by
    # classical capacity.
    c = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg_l2,
                      name="cls_conv1")(inp)
    c = layers.BatchNormalization(name="cls_bn1")(c)
    c = layers.Activation("relu", name="cls_relu1")(c)
    c = layers.MaxPooling2D(2, name="cls_pool1")(c)

    c = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg_l2,
                      name="cls_conv2")(c)
    c = layers.BatchNormalization(name="cls_bn2")(c)
    c = layers.Activation("relu", name="cls_relu2")(c)
    c = layers.GlobalAveragePooling2D(name="cls_gap")(c)   # (B, 64)

    # ── Quantum branch ────────────────────────────────────────────────────
    # MinimalQuantumLayer: 8×8 windows, corner sampling, data re-uploading.
    # Output: (B, 16, 16, N_QUBITS).  PauliZ expectations ∈ [-1, 1].
    q = MinimalQuantumLayer(name="quantum_preprocess")(inp)
    # Shift [-1,1] → [0,1] so the following Conv doesn't receive negative inputs.
    q = layers.Lambda(lambda t: (t + 1.0) / 2.0, name="q_rescale")(q)

    q = layers.Conv2D(16, 3, padding="same", kernel_regularizer=reg_l2,
                      name="q_conv1")(q)
    q = layers.BatchNormalization(name="q_bn1")(q)
    q = layers.LeakyReLU(negative_slope=0.1, name="q_lrelu1")(q)
    q = layers.MaxPooling2D(2, name="q_pool1")(q)

    q = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg_l2,
                      name="q_conv2")(q)
    q = layers.BatchNormalization(name="q_bn2")(q)
    q = layers.Activation("relu", name="q_relu2")(q)
    q = layers.GlobalAveragePooling2D(name="q_gap")(q)     # (B, 32)

    # ── Merge & classify ──────────────────────────────────────────────────
    x = layers.Concatenate(name="concat")([c, q])          # (B, 96)
    x = layers.Dropout(0.5, name="drop1")(x)
    x = layers.Dense(48, activation="gelu",
                     kernel_regularizer=reg_l2, name="fc1")(x)
    x = layers.Dropout(0.4, name="drop2")(x)
    out = layers.Dense(1, activation="sigmoid",
                       kernel_regularizer=reg_l2, name="output")(x)

    return tf.keras.Model(inp, out, name="hybrid_quantum_cnn")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Hybrid Quantum CNN …")
    dummy = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, 1))
    model = create_hybrid_quantum_cnn()
    model.summary()
    pred = model(dummy, training=False)
    print(f"  output shape : {pred.shape}   (expected (2, 1))")
    assert pred.shape == (2, 1), f"Unexpected output shape: {pred.shape}"
    print("✅  Done")
