"""
Quantum Weight Management Utilities
===================================
Functions for saving, loading, and managing quantum circuit weights
independently from the full model.
"""

import os
import numpy as np
import tensorflow as tf
from cnn_model import create_quantum_cnn_classifier


def save_quantum_weights(model, save_path, metadata=None):
    """
    Save quantum weights from a trained model.
    
    Args:
        model: Trained quantum-enhanced CNN model
        save_path: Path to save quantum weights (.npz file)
        metadata: Optional dictionary with additional metadata
    
    Returns:
        dict: Dictionary containing saved weights and metadata
    """
    quantum_weights = [w for w in model.trainable_weights if 'quantum' in w.name.lower()]
    
    if len(quantum_weights) == 0:
        print("⚠ No quantum weights found in model")
        return None
    
    quantum_weight_dict = {}
    
    # Save individual weights
    for i, weight in enumerate(quantum_weights):
        clean_name = weight.name.replace(':', '_').replace('/', '_')
        quantum_weight_dict[f"weight_{i}_{clean_name}"] = weight.numpy()
    
    # Add metadata
    try:
        quantum_layer = model.get_layer('quantum_preprocessing')
        model_metadata = {
            'n_qubits': quantum_layer.n_qubits,
            'n_layers': quantum_layer.n_layers,
            'num_weights': len(quantum_weights),
            'weight_shapes': [w.shape.as_list() for w in quantum_weights],
            'weight_names': [w.name for w in quantum_weights]
        }
    except:
        model_metadata = {
            'num_weights': len(quantum_weights),
            'weight_shapes': [w.shape.as_list() for w in quantum_weights],
            'weight_names': [w.name for w in quantum_weights]
        }
    
    if metadata:
        model_metadata.update(metadata)
    
    quantum_weight_dict['metadata'] = model_metadata
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **quantum_weight_dict)
    
    print(f"✓ Saved {len(quantum_weights)} quantum weights to: {save_path}")
    return quantum_weight_dict


def load_quantum_weights(weights_path):
    """
    Load quantum weights from a saved file.
    
    Args:
        weights_path: Path to the .npz file containing quantum weights
    
    Returns:
        tuple: (weights_list, metadata_dict)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Quantum weights file not found: {weights_path}")
    
    data = np.load(weights_path, allow_pickle=True)
    
    # Extract metadata
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    # Extract weights
    weight_keys = [k for k in data.keys() if k.startswith('weight_') and k != 'metadata']
    weight_keys.sort()  # Ensure consistent order
    
    weights = [data[key] for key in weight_keys]
    
    print(f"✓ Loaded {len(weights)} quantum weights from: {weights_path}")
    print(f"  Metadata: {metadata}")
    
    return weights, metadata


def apply_quantum_weights_to_model(model, weights_list, metadata=None):
    """
    Apply loaded quantum weights to a model.
    
    Args:
        model: Quantum-enhanced CNN model
        weights_list: List of numpy arrays containing quantum weights
        metadata: Optional metadata for verification
    
    Returns:
        bool: True if weights were applied successfully
    """
    quantum_weights = [w for w in model.trainable_weights if 'quantum' in w.name.lower()]
    
    if len(quantum_weights) != len(weights_list):
        print(f"⚠ Weight count mismatch: model has {len(quantum_weights)}, loaded {len(weights_list)}")
        return False
    
    # Verify shapes if metadata is available
    if metadata and 'weight_shapes' in metadata:
        expected_shapes = metadata['weight_shapes']
        for i, (model_weight, loaded_weight) in enumerate(zip(quantum_weights, weights_list)):
            if list(model_weight.shape) != expected_shapes[i]:
                print(f"⚠ Shape mismatch at weight {i}: expected {expected_shapes[i]}, got {model_weight.shape}")
                return False
    
    # Apply weights
    for model_weight, loaded_weight in zip(quantum_weights, weights_list):
        model_weight.assign(loaded_weight)
    
    print(f"✓ Applied {len(weights_list)} quantum weights to model")
    return True


def create_model_with_quantum_weights(weights_path, input_shape=(128, 128, 1)):
    """
    Create a quantum-enhanced CNN model and load pre-trained quantum weights.
    
    Args:
        weights_path: Path to saved quantum weights
        input_shape: Input shape for the model
    
    Returns:
        tf.keras.Model: Model with loaded quantum weights
    """
    # Load weights and metadata
    weights_list, metadata = load_quantum_weights(weights_path)
    
    # Create model with appropriate settings
    n_qubits = metadata.get('n_qubits', 4)
    n_layers = metadata.get('n_layers', 4)
    
    model = create_quantum_cnn_classifier(
        input_shape=input_shape,
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_quantum_preprocessing=True
    )
    
    # Apply quantum weights
    success = apply_quantum_weights_to_model(model, weights_list, metadata)
    
    if success:
        print("✓ Model created with pre-trained quantum weights")
        return model
    else:
        print("⚠ Failed to apply quantum weights, returning model with random initialization")
        return model


def compare_quantum_weights(weights_path1, weights_path2):
    """
    Compare quantum weights from two different files.
    
    Args:
        weights_path1: Path to first quantum weights file
        weights_path2: Path to second quantum weights file
    
    Returns:
        dict: Comparison statistics
    """
    weights1, metadata1 = load_quantum_weights(weights_path1)
    weights2, metadata2 = load_quantum_weights(weights_path2)
    
    if len(weights1) != len(weights2):
        print(f"⚠ Different number of weights: {len(weights1)} vs {len(weights2)}")
        return None
    
    comparison = {}
    
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        if w1.shape != w2.shape:
            print(f"⚠ Shape mismatch at weight {i}: {w1.shape} vs {w2.shape}")
            continue
        
        diff = w1 - w2
        comparison[f'weight_{i}'] = {
            'mse': float(np.mean(diff**2)),
            'max_abs_diff': float(np.max(np.abs(diff))),
            'correlation': float(np.corrcoef(w1.flatten(), w2.flatten())[0, 1])
        }
    
    print(f"✓ Compared {len(weights1)} quantum weight arrays")
    return comparison


# Example usage and testing
if __name__ == "__main__":
    print("Testing Quantum Weight Management...")
    
    # Create a dummy model with quantum weights
    print("\n1. Creating test model...")
    model = create_quantum_cnn_classifier(use_quantum_preprocessing=True)
    
    # Save quantum weights
    print("\n2. Saving quantum weights...")
    test_weights_path = "./test_quantum_weights.npz"
    save_quantum_weights(model, test_weights_path)
    
    # Load quantum weights
    print("\n3. Loading quantum weights...")
    weights, metadata = load_quantum_weights(test_weights_path)
    
    # Create new model and apply weights
    print("\n4. Creating model with loaded weights...")
    new_model = create_model_with_quantum_weights(test_weights_path)
    
    print("\n✓ All quantum weight management tests passed!")
    
    # Cleanup
    if os.path.exists(test_weights_path):
        os.remove(test_weights_path)
        print("✓ Cleaned up test files")