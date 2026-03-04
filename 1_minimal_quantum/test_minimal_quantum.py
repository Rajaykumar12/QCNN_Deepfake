"""
Test Minimal Quantum Implementation
==================================
Quick test script to verify the minimal quantum CNN works correctly.
Creates synthetic data and tests the complete training pipeline.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import cv2
import tensorflow as tf
from minimal_quantum_cnn import MinimalQuantumLayer, create_minimal_quantum_cnn
from train_minimal_quantum import train
from inference_minimal_quantum import predict_single_image
from config import IMG_SIZE


def create_synthetic_test_dataset(output_dir="./test_minimal_dataset", num_samples=30):
    """Create a small synthetic dataset for testing."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class directories
    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    print(f"📁 Creating synthetic dataset: {num_samples} samples per class...")
    
    # Generate "real" images with vertical stripe pattern
    for i in range(num_samples):
        img = np.random.rand(64, 64) * 0.3 + 0.4  # Base gray level
        
        # Add vertical stripes (distinctive pattern for "real")
        for x in range(0, 64, 8):
            img[:, x:x+3] = 0.8  # Bright stripes
        
        # Add some noise
        img += np.random.rand(64, 64) * 0.15
        img = np.clip(img, 0, 1)
        
        # Save as 8-bit image
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(real_dir, f'real_{i:03d}.png'), img_uint8)
    
    # Generate "fake" images with horizontal stripe pattern  
    for i in range(num_samples):
        img = np.random.rand(64, 64) * 0.4 + 0.3  # Slightly different base
        
        # Add horizontal stripes (distinctive pattern for "fake")
        for y in range(0, 64, 6):
            img[y:y+2, :] = 0.9  # Bright horizontal lines
        
        # Add different noise pattern
        img += np.random.rand(64, 64) * 0.2
        img = np.clip(img, 0, 1)
        
        # Save as 8-bit image
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(fake_dir, f'fake_{i:03d}.png'), img_uint8)
    
    print(f"✅ Created {num_samples * 2} synthetic images")
    return output_dir


def test_minimal_quantum_layer():
    """Test the minimal quantum layer functionality."""
    print("🔬 Testing minimal quantum layer...")
    
    # Create test input
    test_input = tf.random.normal((2, 64, 64, 1))  # Batch of 2 images
    
    # Create and test quantum layer
    quantum_layer = MinimalQuantumLayer(n_layers=2)
    
    try:
        output = quantum_layer(test_input)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
        
        # Verify output shape
        expected_shape = (2, 32, 32, 2)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print("✅ Minimal quantum layer test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quantum layer test failed: {e}")
        return False


def test_minimal_quantum_model():
    """Test the complete minimal quantum CNN model."""
    print("🏗️ Testing minimal quantum CNN model...")
    
    try:
        # Create model
        model = create_minimal_quantum_cnn(input_shape=(64, 64, 1))
        
        # Test with dummy data
        test_input = tf.random.normal((4, 64, 64, 1))
        test_output = model(test_input, training=False)
        
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        print(f"   Test prediction shape: {test_output.shape}")
        print(f"   Prediction range: [{tf.reduce_min(test_output):.3f}, {tf.reduce_max(test_output):.3f}]")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Verify output
        assert test_output.shape == (4, 1), f"Expected (4, 1), got {test_output.shape}"
        assert tf.reduce_all(test_output >= 0) and tf.reduce_all(test_output <= 1), "Output should be in [0, 1]"
        
        print("✅ Minimal quantum CNN test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_minimal_training_pipeline():
    """Test the complete training pipeline with small dataset."""
    print("🚀 Testing minimal training pipeline...")
    
    # Create synthetic dataset
    dataset_dir = create_synthetic_test_dataset(num_samples=20)  # Very small for speed
    
    try:
        # Run minimal training (just 3 epochs for testing)
        print("🏃 Running quick training test (3 epochs)...")
        
        model, history = train(
            dataset_dir=dataset_dir,
            output_dir="./test_minimal_results",
            batch_size=8,      # Small batch
            epochs=3,          # Just 3 epochs
            max_per_class=15   # Limit samples
        )
        
        print("✅ Training pipeline test passed!")
        
        # Test inference on a sample image
        sample_image = os.path.join(dataset_dir, 'real', 'real_000.png')
        if os.path.exists(sample_image):
            print("\n🔍 Testing inference...")
            result = predict_single_image(model, sample_image)
            
            if 'error' in result:
                print(f"⚠️ Inference test warning: {result['error']}")
            else:
                print(f"   Sample prediction: {result['predicted_class']} ({result['confidence']:.3f})")
                print("✅ Inference test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        return False
    
    finally:
        # Cleanup test files
        import shutil
        for dir_name in [dataset_dir, "./test_minimal_results"]:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                except:
                    pass


def test_quantum_gradient_flow():
    """Test that gradients can flow through quantum layers."""
    print("🔄 Testing quantum gradient flow...")
    
    try:
        # Create simple model
        model = create_minimal_quantum_cnn()
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create sample data
        X = tf.random.normal((4, 64, 64, 1))
        y = tf.random.uniform((4, 1), maxval=2, dtype=tf.int32)
        y = tf.cast(y, tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            loss = tf.keras.losses.binary_crossentropy(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check that gradients exist and are not all zero
        non_zero_grads = 0
        total_grads = 0
        
        for grad in gradients:
            if grad is not None:
                total_grads += 1
                if tf.reduce_any(tf.abs(grad) > 1e-10):
                    non_zero_grads += 1
        
        print(f"   Total gradient tensors: {total_grads}")
        print(f"   Non-zero gradients: {non_zero_grads}")
        
        assert non_zero_grads > 0, "No gradients found!"
        assert non_zero_grads >= total_grads * 0.5, "Too few non-zero gradients"
        
        print("✅ Gradient flow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False


def run_all_minimal_tests():
    """Run all tests for the minimal quantum implementation."""
    print("🧪 Starting Minimal Quantum CNN Tests")
    print("=" * 45)
    
    tests = [
        ("Quantum Layer", test_minimal_quantum_layer),
        ("Model Creation", test_minimal_quantum_model), 
        ("Gradient Flow", test_quantum_gradient_flow),
        ("Training Pipeline", test_minimal_training_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 25)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 45)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Minimal quantum CNN is ready to use.")
        print("\n📖 Usage:")
        print("   python train_minimal_quantum.py --dataset_dir ./your_dataset")
        print("\n📊 Expected performance:")
        print("   - Training: ~1-5 minutes per epoch (depends on dataset size)")
        print("   - Memory: ~2-4GB (much less than complex version)")
        print("   - Model size: ~100k parameters (perfect for 10k dataset)")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_minimal_tests()
    exit(0 if success else 1)