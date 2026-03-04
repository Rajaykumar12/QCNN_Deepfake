"""
Small Quantum CNN - Reduced backbone to prevent overfitting
===========================================================
Same quantum layer, much smaller CNN (~15k params vs ~100k).
Better suited for limited training samples.

Input:  (batch, 64, 64, 1)
Output: (batch, 1) — probability
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers

from config import IMG_SIZE, N_QUBITS, N_LAYERS


# ---------------------------------------------------------------------------
# Quantum layer (identical to original)
# ---------------------------------------------------------------------------

class MinimalQuantumLayer(tf.keras.layers.Layer):
    """
    2-qubit quantum convolution: (H, W, 1) → (H/2, W/2, 2)
    """

    def __init__(self, n_layers=N_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_qubits = N_QUBITS

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def _circuit(inputs, weights):
            # Encode — one pixel per qubit.
            # RY with π/2 scaling keeps gradients in the high-gradient zone.
            for q in range(self.n_qubits):
                qml.RY(inputs[..., q] * (np.pi / 2), wires=q)

            # Variational layers with ring CNOT
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RX(weights[l, q], wires=q)
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = _circuit

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.n_layers, self.n_qubits),
            initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
            dtype=tf.float32,
            regularizer=tf.keras.regularizers.l2(1e-3),
        )
        super().build(input_shape)

    @tf.function
    def _process_single_image(self, image):
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        oh, ow = h // 2, w // 2

        patches = tf.image.extract_patches(
            tf.expand_dims(image, 0),
            sizes=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.squeeze(patches, 0)
        patches_flat = tf.reshape(patches, (-1, 4))

        qout = self._circuit(patches_flat, self.q_weights)
        qout = tf.stack(qout, axis=0)          # (n_qubits, oh*ow)
        qout = tf.math.real(qout)              # expectation values are real
        qout = tf.cast(qout, tf.float32)
        qout = tf.transpose(qout)              # (oh*ow, n_qubits)

        return tf.reshape(qout, (oh, ow, self.n_qubits))

    def call(self, x, training=None):
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)

        out = tf.map_fn(
            self._process_single_image,
            x,
            fn_output_signature=tf.TensorSpec((None, None, self.n_qubits), tf.float32),
        )
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"n_layers": self.n_layers})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------------
# Small CNN backbone (~15k params instead of ~100k)
# ---------------------------------------------------------------------------

def create_small_quantum_cnn(input_shape=None):
    """
    Minimal quantum layer + tiny CNN backbone.
    
    Architecture:
        64×64×1 → Quantum → 32×32×2
        → Conv(16) → Pool → Conv(32) → GAP → Dense(1)
    
    Total params: ~15k (vs ~100k in the larger version)
    """
    if input_shape is None:
        input_shape = (IMG_SIZE, IMG_SIZE, 1)

    reg = tf.keras.regularizers.l2(1e-4)

    inp = layers.Input(shape=input_shape, name="image_input")

    # Quantum preprocessing
    x = MinimalQuantumLayer(name="quantum_preprocess")(inp)
    # Rescale quantum output [-1, 1] → [0, 1] to prevent dying ReLU.
    x = layers.Lambda(lambda t: (t + 1.0) / 2.0, name="q_rescale")(x)

    # Tiny CNN — only 2 conv layers with fewer filters
    x = layers.Conv2D(16, 3, padding="same", kernel_regularizer=reg, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="lrelu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.3, name="drop1")(x)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="output")(x)

    return tf.keras.Model(inp, out, name="small_quantum_cnn")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Small Quantum CNN …")
    model = create_small_quantum_cnn()
    model.summary()
    
    dummy = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, 1))
    pred = model(dummy)
    print(f"\nInput: {dummy.shape} → Output: {pred.shape}")
    print(f"Total params: {model.count_params():,}")
