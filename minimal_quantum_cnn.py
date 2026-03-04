"""
Minimal Quantum Preprocessing Layer
==================================
Quantum-first preprocessing with spatially-aware patch sampling.
Processes 64×64 grayscale images in the training pipeline.

Input:  (batch, 64, 64, 1) — Raw grayscale images  [0, 1]
Output: (batch, 16, 16, 4) — Quantum feature maps

Key design: each quantum circuit "sees" 4 corner pixels of an 8×8 window
(stride 4), giving a receptive field of 64 pixels² vs the old 4 pixels².
Deepfake artifacts (blending edges, frequency anomalies) require context
across several pixels — the 2×2 patch design was structurally too local.
"""

import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers

from config import IMG_SIZE, N_QUBITS, N_LAYERS


# ---------------------------------------------------------------------------
# Quantum preprocessing layer
# ---------------------------------------------------------------------------

class MinimalQuantumLayer(tf.keras.layers.Layer):
    """
    4-qubit quantum preprocessing layer with spatially-aware sampling.

    Patch strategy: extract 8×8 windows (stride 4) → sample the 4 corner
    pixels → encode one corner per qubit. This gives each circuit call an
    effective receptive field of 8×8 = 64 pixels instead of 4, which is
    necessary for detecting spatially-distributed deepfake artifacts.

    Output shape: (batch, H//4, W//4, 4)  e.g. (B, 16, 16, 4) for 64×64 input.
    Circuit: data re-uploading with qml.Rot (full SU(2)) + ring CNOT per layer.
    Trainable params: N_LAYERS × N_QUBITS × 3 = 36.
    """

    def __init__(self, n_layers=N_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_qubits = N_QUBITS          # always 2

        # --- build the qnode once -------------------------------------------
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def _circuit(inputs, weights):
            """
            Data Re-uploading circuit.

            inputs : (n_patches, n_qubits) — one pixel per qubit
            weights: (n_layers, n_qubits, 3) — Rot gate angles (phi, theta, omega)

            Key design principles:
            1. Re-uploading: input is re-encoded at EVERY layer (not just once).
               Classical circuits encode once; re-uploading interleaves data with
               weights each layer, creating the non-linearity needed for recognition.
            2. arctan encoding: maps [0,1] → [0, π/4], bounded and non-collapsing.
               Avoids zero-gradient zones at 0 and π from simple linear scaling.
            3. qml.Rot (full SU(2)): 3 params per qubit per layer vs 1 in old circuit.
               Shape (N_LAYERS, N_QUBITS, 3) = 36 params vs old 12 — 3× more expressive.
            4. Ring CNOT after each variational block: entangled state changes between
               re-uploadings, preventing the circuit from collapsing to a linear map.
            """
            for l in range(self.n_layers):
                # Re-upload: encode data at the START of every layer.
                # arctan maps pixels in [0,1] → [0, π/4], staying in the
                # high-gradient region of the rotation operators.
                for q in range(self.n_qubits):
                    qml.RY(tf.math.atan(inputs[..., q]), wires=q)
                # Full SU(2) rotation: phi, theta, omega — 3 independent
                # trainable angles per qubit, far more expressive than single RX/RY.
                for q in range(self.n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2],
                            wires=q)
                # Ring CNOT — applied AFTER rotation so entanglement is built on
                # already-variational state, not just raw encoding.
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = _circuit

    # --- Keras weight creation ---------------------------------------------

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            # (layers, qubits, 3) — 3 angles per Rot gate = 36 total params
            # vs old (layers, qubits) = 12 params with single RX/RY.
            shape=(self.n_layers, self.n_qubits, 3),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
            dtype=tf.float32,
            regularizer=tf.keras.regularizers.l2(1e-3),
        )
        super().build(input_shape)

    # --- forward pass ------------------------------------------------------

    @tf.function
    def _process_single_image(self, image):
        """
        Process one (H, W, 1) image → (H//4, W//4, n_qubits) quantum map.

        Patch design (fixes the too-local 2×2 receptive field problem):
          - Extract 8×8 windows with stride 4 using SAME padding.
            → output: (H//4, W//4) spatial positions, each with 64 pixel values.
          - From each 8×8 patch (64 values) take the 4 corner pixels:
            top-left(0), top-right(7), bottom-left(56), bottom-right(63).
          - Each corner is 7 pixels away from its neighbours — the circuit
            now compares pixels across an 8-pixel span per qubit, giving
            sensitivity to blending edges and frequency discontinuities.

        For 64×64 input: output is (16, 16, 4) — 256 circuit calls per image
        (vs 1024 for the old 2×2 design, so also ~4× faster per epoch).
        """
        # Extract 8×8 patches, stride 4, SAME padding → (1, H//4, W//4, 64)
        patches = tf.image.extract_patches(
            tf.expand_dims(image, 0),
            sizes=[1, 8, 8, 1],
            strides=[1, 4, 4, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        patches = tf.squeeze(patches, 0)              # (H//4, W//4, 64)
        oh = tf.shape(patches)[0]                     # 16 for 64×64 input
        ow = tf.shape(patches)[1]
        patches_flat = tf.reshape(patches, (-1, 64))  # (oh*ow, 64)

        # Sample 4 corner pixels from each 8×8 patch.
        # Flat indices in an 8×8 grid: TL=0, TR=7, BL=56, BR=63.
        # Corners are 7 pixels apart — maximum spread within the window.
        corners = tf.constant([0, 7, 56, 63], dtype=tf.int32)
        q_inputs = tf.gather(patches_flat, corners, axis=1)  # (oh*ow, 4)

        # Quantum circuit: returns list of n_qubits tensors each (oh*ow,)
        qout = self._circuit(q_inputs, self.q_weights)
        qout = tf.stack(qout, axis=0)   # (n_qubits, oh*ow)
        qout = tf.math.real(qout)       # expectation values are real; drop ~0 imag
        qout = tf.cast(qout, tf.float32)
        qout = tf.transpose(qout)       # (oh*ow, n_qubits)

        return tf.reshape(qout, (oh, ow, self.n_qubits))

    def call(self, x, training=None):
        """
        x : (batch, H, W, 1)
        returns : (batch, H//4, W//4, n_qubits)  e.g. (B, 16, 16, 4) for 64×64
        """
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)

        # Process each image independently so PennyLane only builds a
        # graph over (H/2)*(W/2) patches at a time.
        out = tf.map_fn(
            self._process_single_image,
            x,
            fn_output_signature=tf.TensorSpec(
                (None, None, self.n_qubits), tf.float32
            ),
        )
        return out

    # --- serialisation (needed for model.save / load) ----------------------

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_layers": self.n_layers})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------------
# Multi-Observable Quantum Layer  (Phase 2 / feature-caching use)
# ---------------------------------------------------------------------------

class MultiObservableQuantumLayer(MinimalQuantumLayer):
    """
    Extends MinimalQuantumLayer to return PauliX + PauliY + PauliZ per qubit.

    Output shape : (batch, H//4, W//4, 3 × N_QUBITS)  →  (B, 16, 16, 12)

    Why three observables?
    ─────────────────────
    PauliZ alone gives one projection of each qubit's Bloch-sphere state.
    X and Y projections from the same circuit capture orthogonal components
    of the quantum state, providing ≈3× more information about the circuit's
    output for the same number of circuit executions.

    Compatible with pre-trained MinimalQuantumLayer weights:
      The gate structure is identical; only the measurement basis changes.
      Loading Z-only trained weights and then measuring X/Y/Z is valid —
      the extra observables reflect the same trained state from new angles.

    Usage (Phase 2 caching):
        layer = MultiObservableQuantumLayer()
        layer.set_weights(pretrained_z_weights)   # load Phase-1 weights
        layer.trainable = False                   # freeze — Phase 3 trains classifier
    """

    def __init__(self, n_layers=N_LAYERS, **kwargs):
        # Call grandparent (tf.keras.layers.Layer), NOT MinimalQuantumLayer.__init__
        # because we need to rebuild the circuit with different measurements.
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.n_layers = n_layers
        self.n_qubits = N_QUBITS
        self._n_out   = 3 * N_QUBITS          # 12 output channels for 4 qubits

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def _multi_circuit(inputs, weights):
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RY(tf.math.atan(inputs[..., q]), wires=q)
                for q in range(self.n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2],
                            wires=q)
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
            # Return X, Y, Z expectations for every qubit — 12 values total
            return (
                [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)] +
                [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)] +
                [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            )

        self._circuit = _multi_circuit

    # Reuse build() from parent (creates q_weights with same shape)

    @tf.function
    def _process_single_image(self, image):
        """Same patch extraction as parent; returns (H//4, W//4, 12)."""
        patches = tf.image.extract_patches(
            tf.expand_dims(image, 0),
            sizes=[1, 8, 8, 1], strides=[1, 4, 4, 1],
            rates=[1, 1, 1, 1], padding="SAME",
        )
        patches    = tf.squeeze(patches, 0)
        oh         = tf.shape(patches)[0]
        ow         = tf.shape(patches)[1]
        patches_flat = tf.reshape(patches, (-1, 64))
        corners    = tf.constant([0, 7, 56, 63], dtype=tf.int32)
        q_inputs   = tf.gather(patches_flat, corners, axis=1)

        qout = self._circuit(q_inputs, self.q_weights)
        qout = tf.stack(qout, axis=0)          # (12, oh*ow)
        qout = tf.math.real(qout)
        qout = tf.cast(qout, tf.float32)
        qout = tf.transpose(qout)              # (oh*ow, 12)
        return tf.reshape(qout, (oh, ow, self._n_out))

    def call(self, x, training=None):
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        return tf.map_fn(
            self._process_single_image, x,
            fn_output_signature=tf.TensorSpec(
                (None, None, self._n_out), tf.float32
            ),
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_layers": self.n_layers})
        return cfg


# ---------------------------------------------------------------------------
# Complete model
# ---------------------------------------------------------------------------

def create_minimal_quantum_cnn(input_shape=None):
    """
    Quantum-first CNN with spatially-aware quantum preprocessing.

    Pipeline:
        64×64×1
        → MinimalQuantumLayer (8×8 windows, corner sampling) → 16×16×4
        → [0,1] rescale
        → Conv(16) → BN → LeakyReLU → Pool(2) → Dropout  →  8×8×16
        → Conv(32) → BN → ReLU → GAP                      →  32
        → Dropout → Dense(1, sigmoid)

    Capacity breakdown (why small CNN body matters):
        Quantum (q_weights): 36 params   (N_LAYERS=3 × N_QUBITS=4 × 3 angles)
        CNN body:           ~5 400 params
        Quantum fraction:   ~0.66%  vs 0.01% in the original large-CNN design
    """
    if input_shape is None:
        input_shape = (IMG_SIZE, IMG_SIZE, 1)

    reg = tf.keras.regularizers.l2(1e-4)
    inp = layers.Input(shape=input_shape, name="image_input")

    # Quantum preprocessing — data re-uploading, 36 trainable params
    x = MinimalQuantumLayer(name="quantum_preprocess")(inp)
    # PauliZ expectation values are in [-1, 1]; shift to [0, 1] so the
    # first Conv layer receives non-negative inputs (avoids dying ReLU).
    x = layers.Lambda(lambda t: (t + 1.0) / 2.0, name="q_rescale")(x)

    # Intentionally small CNN body — forces the model to use quantum features.
    # Old body (Conv32→Conv64→Conv128) had 94k params; quantum (12) was 0.01%.
    # New body has ~5k params; quantum (36) is ~0.7% — 70× more influence.
    x = layers.Conv2D(16, 3, padding="same", kernel_regularizer=reg, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.LeakyReLU(negative_slope=0.1, name="lrelu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dropout(0.4, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="output")(x)

    return tf.keras.Model(inp, out, name="minimal_quantum_cnn")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing MinimalQuantumLayer …")
    dummy = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, 1))
    layer = MinimalQuantumLayer()
    y = layer(dummy)
    print(f"  in  {dummy.shape}  →  out  {y.shape}")
    assert y.shape == (2, IMG_SIZE // 4, IMG_SIZE // 4, N_QUBITS), \
        f"Unexpected output shape: {y.shape}"

    print("\nTesting full model …")
    model = create_minimal_quantum_cnn()
    model.summary()
    pred = model(dummy)
    print(f"  predictions shape: {pred.shape}")
    print("✅  Done")