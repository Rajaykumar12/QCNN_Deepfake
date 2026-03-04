"""
Quantum CNN Inference Script
============================
Demonstrates how to use saved quantum weights for inference on new images.

Usage:
    python inference_quantum_cnn.py --quantum_weights ./content/quantum_weights.npz \
                                    --image_path ./test_image.jpg
    
    python inference_quantum_cnn.py --model_path ./content/quantum_cnn_model.keras \
                                    --image_path ./test_image.jpg
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from cnn_model import create_quantum_cnn_classifier
from quantum_weights_utils import create_model_with_quantum_weights


def preprocess_single_image(image_path, target_size=(128, 128)):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width)
    
    Returns:
        np.array: Preprocessed image ready for model input
    """
    try:
        # Load and convert to grayscale
        image = Image.open(image_path).convert('L')
        
        # Resize to target size
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch and channel dimensions: (1, 128, 128, 1)
        image_array = image_array[np.newaxis, ..., np.newaxis]
        
        return image_array
    
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None


def predict_single_image(model, image_path, threshold=0.5):
    """
    Predict whether a single image is real or fake.
    
    Args:
        model: Trained quantum CNN model
        image_path: Path to the image file
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: Prediction results
    """
    # Preprocess image
    image_array = preprocess_single_image(image_path)
    if image_array is None:
        return None
    
    # Get prediction
    prediction_prob = model.predict(image_array, verbose=0)[0, 0]
    prediction_binary = int(prediction_prob > threshold)
    confidence = max(prediction_prob, 1 - prediction_prob)
    
    # Interpret results
    if prediction_binary == 0:
        prediction_label = "REAL"
        confidence_direction = f"{(1 - prediction_prob) * 100:.1f}% confident it's REAL"
    else:
        prediction_label = "FAKE"
        confidence_direction = f"{prediction_prob * 100:.1f}% confident it's FAKE"
    
    return {
        'image_path': image_path,
        'probability_fake': float(prediction_prob),
        'prediction_binary': prediction_binary,
        'prediction_label': prediction_label,
        'confidence': float(confidence),
        'confidence_description': confidence_direction,
        'threshold_used': threshold
    }


def batch_predict_directory(model, directory_path, threshold=0.5):
    """
    Predict on all images in a directory.
    
    Args:
        model: Trained quantum CNN model
        directory_path: Path to directory containing images
        threshold: Classification threshold
    
    Returns:
        list: List of prediction results
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    results = []
    
    print(f"\n📂 Processing images from directory: {directory_path}")
    
    image_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        result = predict_single_image(model, image_path, threshold)
        if result:
            results.append(result)
            print(f"  → {result['prediction_label']} ({result['confidence_description']})")
        else:
            print(f"  → Error processing image")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="Path to saved complete model (.keras)")
    parser.add_argument("--quantum_weights", default=None,
                        help="Path to quantum weights file (.npz)")
    parser.add_argument("--image_path", default=None,
                        help="Path to single image file")
    parser.add_argument("--directory_path", default=None,
                        help="Path to directory containing images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--output_file", default=None,
                        help="Optional: save results to JSON file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model_path and not args.quantum_weights:
        print("❌ Error: Must provide either --model_path or --quantum_weights")
        return
    
    if not args.image_path and not args.directory_path:
        print("❌ Error: Must provide either --image_path or --directory_path")
        return
    
    print("=" * 60)
    print("QUANTUM CNN INFERENCE")
    print("=" * 60)
    
    # Load model
    if args.model_path:
        print(f"\n🤖 Loading saved model: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
        
        # Check for quantum components
        quantum_layers = [l for l in model.layers if 'quantum' in l.name.lower()]
        if quantum_layers:
            print(f"  ✓ Loaded quantum-enhanced model with {len(quantum_layers)} quantum layers")
        else:
            print("  ⚠ Loaded classical model (no quantum layers detected)")
    
    elif args.quantum_weights:
        print(f"\n🔬 Creating model with quantum weights: {args.quantum_weights}")
        model = create_model_with_quantum_weights(args.quantum_weights)
        
        print("🔒 INFERENCE MODE: Quantum weights are FROZEN")
        print("   ✓ No parameter updates during inference")
        print("   ✓ Deterministic, reproducible predictions")
        print("   ✓ Using trained quantum features for classification")
        
        from quantum_weights_utils import load_quantum_weights
        _, metadata = load_quantum_weights(args.quantum_weights)
        print(f"   Quantum configuration: {metadata.get('n_qubits', 'unknown')} qubits, "
              f"{metadata.get('n_layers', 'unknown')} layers")
    
    print(f"Classification threshold: {args.threshold}")
    
    # Run inference
    results = []
    
    if args.image_path:
        # Single image prediction
        print(f"\n🖼️ Analyzing single image: {args.image_path}")
        result = predict_single_image(model, args.image_path, args.threshold)
        
        if result:
            results.append(result)
            print(f"\n📊 Results:")
            print(f"  Image: {result['image_path']}")
            print(f"  Prediction: {result['prediction_label']}")
            print(f"  Confidence: {result['confidence_description']}")
            print(f"  Raw probability (fake): {result['probability_fake']:.4f}")
        else:
            print("❌ Failed to process image")
    
    elif args.directory_path:
        # Batch prediction
        results = batch_predict_directory(model, args.directory_path, args.threshold)
        
        # Summary statistics
        if results:
            total_images = len(results)
            fake_predictions = sum(1 for r in results if r['prediction_binary'] == 1)
            real_predictions = total_images - fake_predictions
            avg_confidence = sum(r['confidence'] for r in results) / total_images
            
            print(f"\n📊 Summary:")
            print(f"  Total images processed: {total_images}")
            print(f"  Predicted as REAL: {real_predictions}")
            print(f"  Predicted as FAKE: {fake_predictions}")
            print(f"  Average confidence: {avg_confidence:.3f}")
    
    # Save results if requested
    if args.output_file and results:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_file}")
    
    print(f"\n✓ Inference complete!")


if __name__ == "__main__":
    main()