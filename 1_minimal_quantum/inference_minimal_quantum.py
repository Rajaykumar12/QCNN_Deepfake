"""
Minimal Quantum CNN Inference
=============================
Inference script for the minimal quantum CNN.

Usage:
    python inference_minimal_quantum.py --model_path ./model.keras --image_path ./test.jpg
    python inference_minimal_quantum.py --model_path ./model.keras --dataset_dir ./test_dataset
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from config import CLASSES, IMG_SIZE
from minimal_quantum_cnn import MinimalQuantumLayer   # needed for custom_objects


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """Load and preprocess a single image for inference.

    Returns array of shape (1, img_size, img_size, 1) or None on error.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, ..., np.newaxis]      # (1, H, W, 1)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def predict_single_image(model, image_path, threshold=0.5):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to image
        threshold: Decision threshold
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return {'error': 'Could not load image'}
    
    try:
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Convert to class prediction
        predicted_class_idx = 1 if prediction > threshold else 0
        predicted_class = CLASSES[predicted_class_idx]
        confidence = prediction if predicted_class_idx == 1 else (1 - prediction)
        
        return {
            'image_path': image_path,
            'prediction_score': float(prediction),
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': float(confidence),
            'is_fake': predicted_class_idx == 1,
            'threshold_used': threshold
        }
        
    except Exception as e:
        return {
            'image_path': image_path,
            'error': f'Prediction failed: {e}'
        }


def predict_dataset_directory(model, dataset_dir, threshold=0.5, max_images_per_class=None):
    """
    Make predictions on all images in dataset directory.
    
    Args:
        model: Trained model
        dataset_dir: Directory with class subdirs or just images
        threshold: Decision threshold
        max_images_per_class: Limit images per class
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    print(f"🔍 Processing directory: {dataset_dir}")
    
    # Check if directory has class structure
    class_dirs = []
    for class_name in CLASSES:
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_path):
            class_dirs.append((class_name, class_path))
    
    if class_dirs:
        # Process class subdirectories
        for class_name, class_path in class_dirs:
            print(f"📁 Processing {class_name} class...")
            
            class_results = process_directory_images(
                model, class_path, threshold, 
                true_class=class_name,
                max_images=max_images_per_class
            )
            results.extend(class_results)
            print(f"  Processed {len(class_results)} images")
    
    else:
        # Process single directory
        print("📁 Processing single directory (no class structure)...")
        dir_results = process_directory_images(
            model, dataset_dir, threshold,
            max_images=max_images_per_class
        )
        results.extend(dir_results)
    
    print(f"✅ Total processed: {len(results)} images")
    return results


def process_directory_images(model, directory, threshold, true_class=None, max_images=None):
    """Process all images in a directory."""
    results = []
    image_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if max_images and image_count >= max_images:
                break
                
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                # Make prediction
                result = predict_single_image(model, img_path, threshold)
                
                # Add true class if known
                if true_class:
                    result['true_class'] = true_class
                    result['true_class_idx'] = CLASSES.index(true_class)
                
                results.append(result)
                image_count += 1
        
        if max_images and image_count >= max_images:
            break
    
    return results


def calculate_accuracy_metrics(results):
    """Calculate accuracy metrics from results."""
    if not results:
        return {}
    
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r and 'true_class' in r]
    
    if not valid_results:
        return {'error': 'No valid predictions with true labels'}
    
    # Overall accuracy
    correct = sum(1 for r in valid_results 
                 if r['predicted_class_idx'] == r['true_class_idx'])
    total = len(valid_results)
    overall_accuracy = correct / total
    
    # Per-class accuracy
    class_metrics = {}
    for class_idx, class_name in enumerate(CLASSES):
        class_results = [r for r in valid_results if r['true_class_idx'] == class_idx]
        if class_results:
            class_correct = sum(1 for r in class_results 
                              if r['predicted_class_idx'] == r['true_class_idx'])
            class_accuracy = class_correct / len(class_results)
            class_metrics[class_name] = {
                'accuracy': class_accuracy,
                'total': len(class_results),
                'correct': class_correct
            }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_predictions': total,
        'correct_predictions': correct,
        'class_metrics': class_metrics,
        'error_count': len(results) - len(valid_results)
    }


def main():
    parser = argparse.ArgumentParser(description='Minimal Quantum CNN Inference')
    parser.add_argument('--model_path', required=True,
                       help='Path to trained .keras model file')
    parser.add_argument('--image_path',
                       help='Path to single image for prediction')
    parser.add_argument('--dataset_dir', 
                       help='Path to directory with images for batch prediction')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for binary classification')
    parser.add_argument('--max_images', type=int,
                       help='Maximum images to process per class')
    parser.add_argument('--output_csv',
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.dataset_dir:
        parser.error("Please provide either --image_path or --dataset_dir")
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Load model (register custom quantum layer)
    print(f"Loading model from: {args.model_path}")
    try:
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={"MinimalQuantumLayer": MinimalQuantumLayer},
        )
        print(f"Model loaded  |  input {model.input_shape}  output {model.output_shape}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Single image prediction
    if args.image_path:
        print(f"\n🔍 Predicting single image: {args.image_path}")
        
        if not os.path.exists(args.image_path):
            print(f"❌ Image file not found: {args.image_path}")
            return
        
        result = predict_single_image(model, args.image_path, args.threshold)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n📊 Prediction Results:")
            print(f"   Image: {result['image_path']}")
            print(f"   Predicted Class: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Raw Score: {result['prediction_score']:.4f}")
            print(f"   Is Fake: {'Yes' if result['is_fake'] else 'No'}")
    
    # Dataset prediction
    if args.dataset_dir:
        print(f"\n🔍 Predicting dataset: {args.dataset_dir}")
        
        if not os.path.exists(args.dataset_dir):
            print(f"❌ Dataset directory not found: {args.dataset_dir}")
            return
        
        results = predict_dataset_directory(
            model, args.dataset_dir, 
            args.threshold, args.max_images
        )
        
        if not results:
            print("❌ No images processed")
            return
        
        # Calculate metrics if we have true labels
        metrics = calculate_accuracy_metrics(results)
        
        if 'error' in metrics:
            print(f"⚠️ {metrics['error']}")
        else:
            print(f"\n📊 Dataset Results:")
            print(f"   Overall Accuracy: {metrics['overall_accuracy']:.3f}")
            print(f"   Total Images: {metrics['total_predictions']}")
            print(f"   Correct Predictions: {metrics['correct_predictions']}")
            
            if 'class_metrics' in metrics:
                print(f"\n📈 Per-Class Results:")
                for class_name, class_data in metrics['class_metrics'].items():
                    print(f"   {class_name}: {class_data['accuracy']:.3f} "
                          f"({class_data['correct']}/{class_data['total']})")
        
        # Save to CSV if requested
        if args.output_csv:
            try:
                import pandas as pd
                
                # Prepare data for CSV
                csv_data = []
                for result in results:
                    if 'error' not in result:
                        csv_data.append({
                            'image_path': result['image_path'],
                            'predicted_class': result['predicted_class'],
                            'prediction_score': result['prediction_score'],
                            'confidence': result['confidence'],
                            'is_fake': result['is_fake'],
                            'true_class': result.get('true_class', 'unknown')
                        })
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    df.to_csv(args.output_csv, index=False)
                    print(f"💾 Results saved to: {args.output_csv}")
                else:
                    print("⚠️ No valid results to save")
                    
            except ImportError:
                print("⚠️ pandas not available, skipping CSV export")
            except Exception as e:
                print(f"❌ Failed to save CSV: {e}")


if __name__ == "__main__":
    main()