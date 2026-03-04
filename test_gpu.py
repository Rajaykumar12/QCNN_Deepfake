#!/usr/bin/env python3
"""Test GPU availability for TensorFlow"""

import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully")
    print(f"  Version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n✓ GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    if gpus:
        print("\n✓ GPU available - training will be fast!")
        print("  Initializing GPU memory...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  Memory growth enabled")
    else:
        print("\n✗ No GPUs detected - falling back to CPU")
        
except Exception as e:
    print(f"✗ Error importing TensorFlow: {e}")
    import traceback
    traceback.print_exc()
