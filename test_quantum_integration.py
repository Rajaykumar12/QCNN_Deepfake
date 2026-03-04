"""
Test Script for Quantum-Enhanced CNN
====================================
Verifies that the quantum preprocessing integration works correctly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from cnn_model import create_quantum_cnn_classifier
from quantum_preprocessing import DualQuantumPreprocessor, create_raw_image_loader

def test_quantum_preprocessing():
    """Test quantum preprocessing layer independently."""
    print("🔬 Testing Quantum Preprocessing Layer...")
    
    # Create test input (raw grayscale images)
    test_images = tf.random.uniform((4, 128, 128, 1), dtype=tf.float32)
    
    # Create quantum preprocessor
    quantum_prep = DualQuantumPreprocessor(n_qubits=4, n_layers=2)
    
    # Process through quantum layer
    quantum_features = quantum_prep(test_images)
    
    print(f"  Input shape: {test_images.shape}")
    print(f"  Output shape: {quantum_features.shape}")
    print(f"  Trainable weights: {len(quantum_prep.trainable_weights)}")
    
    # Verify output characteristics
    assert quantum_features.shape[0] == 4  # Batch size
    assert quantum_features.shape[1] == 64  # Height // 2
    assert quantum_features.shape[2] == 64  # Width // 2
    assert quantum_features.shape[3] == 8   # n_qubits * 2
    
    print("  ✓ Quantum preprocessing test passed!")
    return True

def test_quantum_cnn_model():
    """Test the full quantum-enhanced CNN model."""
    print("\n🧠 Testing Quantum-Enhanced CNN Model...")
    
    # Create model with quantum preprocessing
    model = create_quantum_cnn_classifier(
        input_shape=(128, 128, 1),
        use_quantum_preprocessing=True
    )
    
    # Test input (raw grayscale images)
    test_input = np.random.rand(2, 128, 128, 1).astype(np.float32)
    
    # Forward pass
    predictions = model.predict(test_input, verbose=0)
    
    print(f"  Model input shape: {test_input.shape}")
    print(f"  Model output shape: {predictions.shape}")
    print(f"  Output values: {predictions.flatten()}")
    
    # Count parameters
    quantum_weights = [w for w in model.trainable_weights if 'quantum' in w.name.lower()]
    classical_weights = [w for w in model.trainable_weights if 'quantum' not in w.name.lower()]
    
    print(f"  Quantum parameters: {len(quantum_weights)}")
    print(f"  Classical parameters: {len(classical_weights)}")
    print(f"  Total parameters: {len(model.trainable_weights)}")
    
    # Verify output characteristics
    assert predictions.shape == (2, 1)  # Batch size x 1 output
    assert np.all(predictions >= 0) and np.all(predictions <= 1)  # Sigmoid output
    assert len(quantum_weights) > 0  # Should have quantum parameters
    
    print("  ✓ Quantum CNN model test passed!")
    return True

def test_classical_mode():
    """Test the model in classical mode for comparison."""
    print("\n🔧 Testing Classical Mode...")
    
    # Create model without quantum preprocessing
    model = create_quantum_cnn_classifier(
        input_shape=(128, 128, 1),
        use_quantum_preprocessing=False
    )
    
    # Test input
    test_input = np.random.rand(2, 128, 128, 1).astype(np.float32)
    predictions = model.predict(test_input, verbose=0)
    
    # Count parameters
    quantum_weights = [w for w in model.trainable_weights if 'quantum' in w.name.lower()]
    classical_weights = [w for w in model.trainable_weights if 'quantum' not in w.name.lower()]
    
    print(f"  Model output shape: {predictions.shape}")
    print(f"  Quantum parameters: {len(quantum_weights)}")
    print(f"  Classical parameters: {len(classical_weights)}")
    
    # In classical mode, should have no quantum parameters
    assert len(quantum_weights) == 0
    assert predictions.shape == (2, 1)
    
    print("  ✓ Classical mode test passed!")
    return True

def test_gradients():
    """Test that gradients flow through the quantum layers."""
    print("\n🔄 Testing Gradient Flow...")
    
    # Create simple model for gradient testing
    model = create_quantum_cnn_classifier(
        input_shape=(128, 128, 1),
        use_quantum_preprocessing=True
    )
    
    # Compile with a simple loss for testing
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy data
    x = np.random.rand(4, 128, 128, 1).astype(np.float32)
    y = np.random.randint(0, 2, (4, 1)).astype(np.float32)
    
    # Test that we can compute gradients
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_weights)
    
    # Check that quantum parameters have gradients
    quantum_grads = [g for g, w in zip(gradients, model.trainable_weights) 
                    if 'quantum' in w.name.lower() and g is not None]
    
    print(f"  Total gradients: {len([g for g in gradients if g is not None])}")
    print(f"  Quantum gradients: {len(quantum_grads)}")
    
    assert len(quantum_grads) > 0, "Quantum parameters should have gradients!"
    
    print("  ✓ Gradient flow test passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("QUANTUM-ENHANCED CNN INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Run all tests
        test_quantum_preprocessing()
        test_quantum_cnn_model()
        test_classical_mode()
        test_gradients()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✓ Quantum preprocessing integration is working correctly")
        print("✓ End-to-end backpropagation through quantum layers is functional")
        print("✓ Both quantum and classical modes are operational")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()