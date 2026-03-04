"""
QViT Deepfake Detection — Quantum-Enhanced CNN Classifier
===========================================================
Enhanced CNN with integrated PennyLane quantum preprocessing:
    • End-to-end trainable quantum image preprocessing
    • Residual connections for better feature learning
    • Spatial attention mechanisms 
    • Progressive feature extraction
    • Quantum + classical hybrid architecture

Architecture:
    • Quantum preprocessing layer (trainable)
    • 6 ResNet-style blocks with increasing channels
    • Spatial attention layers
    • Multi-scale feature fusion
    • Improved classification head
    Total: ~3.5M + quantum parameters
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras import layers

from config import IMG_SIZE, HEAD_DROPOUT, N_QUBITS, N_LAYERS
from quantum_preprocessing import DualQuantumPreprocessor


def spatial_attention_block(inputs, name_prefix):
    """
    Spatial attention mechanism to focus on important regions.
    """
    # Channel-wise global pooling
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name_prefix}_avg_pool")(inputs)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True, name=f"{name_prefix}_max_pool")(inputs)
    
    # Attention weights
    concat = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])
    attention = layers.Conv2D(1, 1, activation='sigmoid', padding='same', name=f"{name_prefix}_attention")(concat)
    
    # Apply attention
    return layers.Multiply(name=f"{name_prefix}_attended")([inputs, attention])


def residual_block(inputs, filters, stride=1, name_prefix=""):
    """
    Residual block with batch normalization and optional skip connection.
    """
    # Main path
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, 
                     name=f"{name_prefix}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)
    x = layers.Dropout(0.1, name=f"{name_prefix}_drop1")(x)
    
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False, 
                     name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    
    # Skip connection
    if stride != 1 or inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, 1, strides=stride, use_bias=False, 
                              name=f"{name_prefix}_skip")(inputs)
        inputs = layers.BatchNormalization(name=f"{name_prefix}_skip_bn")(inputs)
    
    x = layers.Add(name=f"{name_prefix}_add")([x, inputs])
    x = layers.Activation('relu', name=f"{name_prefix}_relu2")(x)
    
    return x


def create_quantum_cnn_classifier(
    input_shape=None,
    head_dropout=HEAD_DROPOUT,
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    use_quantum_preprocessing=True
):
    """
    Build quantum-enhanced CNN classifier with integrated PennyLane preprocessing.

    Input:  (batch, 128, 128, 1)  raw grayscale images [0, 1]
    Output: (batch, 1)            probability of being fake
    
    The model integrates:
    1. Quantum preprocessing (trainable PennyLane circuits)
    2. Enhanced CNN with residual blocks and attention
    3. End-to-end backpropagation through quantum + classical layers
    """
    if input_shape is None:
        input_shape = (IMG_SIZE, IMG_SIZE, 1)  # Raw grayscale images

    inputs = layers.Input(shape=input_shape, name="raw_image_input")
    
    # ── Quantum Preprocessing Layer ──────────────────────
    if use_quantum_preprocessing:
        print("🔬 Adding quantum preprocessing layer...")
        x = DualQuantumPreprocessor(
            n_qubits=n_qubits, 
            n_layers=n_layers, 
            name="quantum_preprocessing"
        )(inputs)  # (batch, 64, 64, n_qubits*2)
        
        # Adapt quantum features for CNN processing
        x = layers.Conv2D(64, 1, padding='same', use_bias=False, name="quantum_adapter")(x)
        x = layers.BatchNormalization(name="quantum_adapter_bn")(x)
        x = layers.Activation('relu', name="quantum_adapter_relu")(x)
    else:
        # Traditional preprocessing (for comparison)
        x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name="initial_conv")(inputs)
        x = layers.BatchNormalization(name="initial_bn")(x)
        x = layers.Activation('relu', name="initial_relu")(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same', name="initial_pool")(x)
    
    # ── Residual blocks with progressive channels ─
    x = residual_block(x, 64, stride=1, name_prefix="res1")
    x = spatial_attention_block(x, "att1")
    
    x = residual_block(x, 128, stride=2, name_prefix="res2")  # 16x16x128
    x = spatial_attention_block(x, "att2")
    
    x = residual_block(x, 256, stride=2, name_prefix="res3")  # 8x8x256
    x = residual_block(x, 256, stride=1, name_prefix="res4")
    x = spatial_attention_block(x, "att3")
    
    x = residual_block(x, 512, stride=2, name_prefix="res5")  # 4x4x512
    x = residual_block(x, 512, stride=1, name_prefix="res6")
    x = spatial_attention_block(x, "att4")
    
    # ── Multi-scale feature fusion ───────────────
    # Global Average Pooling
    gap = layers.GlobalAveragePooling2D(name="gap")(x)  # 512
    
    # Global Max Pooling for complementary features
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)  # 512
    
    # Combine features
    features = layers.Concatenate(name="feature_concat")([gap, gmp])  # 1024
    
    # ── Enhanced classification head ──────────────
    x = layers.Dense(512, name="head_dense1")(features)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Activation('relu', name="head_relu1")(x)
    x = layers.Dropout(head_dropout, name="head_dropout1")(x)
    
    x = layers.Dense(256, name="head_dense2")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Activation('relu', name="head_relu2")(x)
    x = layers.Dropout(head_dropout * 0.5, name="head_dropout2")(x)
    
    x = layers.Dense(128, name="head_dense3")(x)
    x = layers.BatchNormalization(name="head_bn3")(x)
    x = layers.Activation('relu', name="head_relu3")(x)
    x = layers.Dropout(head_dropout * 0.3, name="head_dropout3")(x)
    
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model_name = "quantum_cnn_classifier" if use_quantum_preprocessing else "enhanced_cnn_classifier"
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# Backward compatibility
def create_cnn_classifier(**kwargs):
    """Backward compatible function - now creates quantum-enhanced CNN by default."""
    return create_quantum_cnn_classifier(**kwargs)


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Test quantum-enhanced model
    print("Testing Quantum-Enhanced CNN...")
    model = create_quantum_cnn_classifier(use_quantum_preprocessing=True)
    model.summary()

    import numpy as np
    dummy = np.random.rand(2, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)  # Raw grayscale
    out = model.predict(dummy, verbose=0)
    print(f"\nDummy input:  {dummy.shape}")
    print(f"Model output: {out.shape}  values: {out.flatten()}")
    
    print(f"\nQuantum parameters: {sum(1 for w in model.trainable_weights if 'quantum' in w.name)}")
    print(f"Total trainable weights: {len(model.trainable_weights)}")
