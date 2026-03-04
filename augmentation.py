"""
Enhanced Data Augmentation for Deepfake Detection
=================================================
TensorFlow 2.x compatible augmentation for dual-channel deepfake detection.
"""

import tensorflow as tf
import numpy as np


# TensorFlow 2.x compatible version
def create_augmentation_layer():
    """
    Create a Keras preprocessing layer for augmentation.
    Compatible with TensorFlow 2.x - applies to grayscale channel only.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05, fill_mode='constant', fill_value=0.0),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),
        tf.keras.layers.RandomBrightness(0.1, value_range=[0, 1]),
        tf.keras.layers.RandomContrast(0.1),
    ], name='augmentation')


def augment_dual_channel_data(images, labels, augmentation_layer):
    """
    Apply augmentation to dual-channel data while preserving DCT features.
    
    Args:
        images: (batch, height, width, 2) tensor
        labels: (batch,) tensor
        augmentation_layer: Keras preprocessing layer
    
    Returns:
        Augmented (images, labels) tuple
    """
    # Separate channels
    gray_channel = images[:, :, :, 0:1]  # (batch, h, w, 1)
    dct_channel = images[:, :, :, 1:2]   # (batch, h, w, 1)
    
    # Apply augmentation only to grayscale channel
    # Convert to 3 channels for compatibility with standard layers
    gray_3ch = tf.repeat(gray_channel, 3, axis=-1)
    augmented_gray_3ch = augmentation_layer(gray_3ch, training=True)
    augmented_gray = augmented_gray_3ch[:, :, :, 0:1]  # Take first channel back
    
    # Apply the same geometric transformations to DCT channel
    # by using a simpler approach - just the brightness/contrast won't affect DCT
    dct_3ch = tf.repeat(dct_channel, 3, axis=-1)
    # Only apply geometric transformations to DCT (no color changes)
    geometric_only = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05, fill_mode='constant', fill_value=0.0),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),
    ])
    augmented_dct_3ch = geometric_only(dct_3ch, training=True)
    augmented_dct = augmented_dct_3ch[:, :, :, 0:1]
    
    # Combine channels
    augmented_images = tf.concat([augmented_gray, augmented_dct], axis=-1)
    
    return augmented_images, labels