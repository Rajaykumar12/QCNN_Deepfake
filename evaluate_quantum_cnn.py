"""
Quantum-Enhanced CNN Evaluation Script
======================================
Evaluates the trained quantum-enhanced CNN on test data, with support
for loading pre-trained quantum weights.

Usage:
    # Evaluate with saved model
    python evaluate_quantum_cnn.py --model_path ./content/quantum_cnn_model.keras \
                                   --test_dir ./Deepfake_Dataset/Test
    
    # Evaluate with separate quantum weights  
    python evaluate_quantum_cnn.py --quantum_weights ./content/quantum_weights.npz \
                                   --test_dir ./Deepfake_Dataset/Test
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve,
)

from config import CLASSES
from cnn_model import create_quantum_cnn_classifier
from quantum_preprocessing import create_raw_image_loader
from quantum_weights_utils import create_model_with_quantum_weights, load_quantum_weights


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Quantum CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_roc_curve(fpr, tpr, auc_score, save_path, model_type="Quantum CNN"):
    """Plot and save ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"{model_type}  (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_type}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_probability_distribution(probs, labels, save_path, model_type="Quantum CNN"):
    """Plot prediction probability distribution."""
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(real_probs, bins=40, alpha=0.6, color="#4CAF50",
            label=f"Real  (n={len(real_probs)})", density=True)
    ax.hist(fake_probs, bins=40, alpha=0.6, color="#F44336",
            label=f"Fake  (n={len(fake_probs)})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability (Fake)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Prediction Distribution — {model_type}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def evaluate_model(model, test_images, test_labels, output_dir, model_type="Quantum CNN"):
    """
    Comprehensive evaluation of the model.
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print(f"\n📊 Evaluating {model_type}...")
    
    # Get predictions
    print("  Computing predictions...")
    predictions_prob = model.predict(test_images, verbose=1)
    predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
    test_labels_flat = test_labels.flatten()
    predictions_prob_flat = predictions_prob.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels_flat, predictions_binary)
    precision = precision_score(test_labels_flat, predictions_binary)
    recall = recall_score(test_labels_flat, predictions_binary)
    f1 = f1_score(test_labels_flat, predictions_binary)
    auc = roc_auc_score(test_labels_flat, predictions_prob_flat)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels_flat, predictions_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\n📈 {model_type} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn}, FP: {fp}")
    print(f"    FN: {fn}, TP: {tp}")
    
    # Generate plots
    print(f"\n📊 Generating evaluation plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix plot
    plot_confusion_matrix(cm, CLASSES, 
                         os.path.join(output_dir, "confusion_matrix.png"))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_labels_flat, predictions_prob_flat)
    plot_roc_curve(fpr, tpr, auc, 
                   os.path.join(output_dir, "roc_curve.png"), model_type)
    
    # Probability distribution
    plot_probability_distribution(predictions_prob_flat, test_labels_flat,
                                 os.path.join(output_dir, "probability_distribution.png"), 
                                 model_type)
    
    # Compile results
    results = {
        "model_type": model_type.lower().replace(" ", "_"),
        "accuracy": float(accuracy),
        "precision": float(precision), 
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "test_samples": int(len(test_labels_flat))
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="Path to saved model (.keras)")
    parser.add_argument("--quantum_weights", default=None,
                        help="Path to quantum weights file (.npz)")
    parser.add_argument("--test_dir", default=None,
                        help="Path to test images directory")
    parser.add_argument("--dataset_dir", default=None,
                        help="Path to dataset root (auto-detects Test structure)")
    parser.add_argument("--output_dir", default="./evaluation_results",
                        help="Where to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for prediction")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model_path and not args.quantum_weights:
        print("❌ Error: Must provide either --model_path or --quantum_weights")
        return
    
    if not args.test_dir and not args.dataset_dir:
        print("❌ Error: Must provide either --test_dir or --dataset_dir")
        return
    
    print("=" * 60)
    print("QUANTUM-ENHANCED CNN EVALUATION")
    print("=" * 60)
    
    # ── Flexible Data Loading ──────────────────────────────────
    print(f"\n📂 Loading test data...")
    
    if args.dataset_dir:
        # Auto-detect structure from dataset root
        from quantum_preprocessing import auto_detect_dataset_structure
        structure_info = auto_detect_dataset_structure(args.dataset_dir)
        print(f"Auto-detected dataset structure: {structure_info}")
        
        if structure_info['has_train_test_structure'] and structure_info['test_dir']:
            # Found Test directory
            test_dir = structure_info['test_dir']
            print(f"Using Test directory: {test_dir}")
            
        else:
            # No separate test directory found
            print("❌ Error: No Test directory found in dataset structure")
            print("Available directories:", structure_info['data_dirs'])
            return
    
    elif args.test_dir:
        # Manual test directory specification
        test_dir = args.test_dir
        print(f"Using manual test directory: {test_dir}")
    
    # Load test data
    test_images, test_labels = create_raw_image_loader(
        test_dir, classes=CLASSES, img_size=128
    )
    
    print(f"  Test images: {test_images.shape}")
    print(f"  Test labels: {test_labels.shape}")
    print(f"  Classes: {CLASSES}")
    
    # Class distribution
    unique, counts = np.unique(test_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    {CLASSES[int(cls)]}: {count} images")
    
    # Load or create model
    if args.model_path:
        print(f"\n🤖 Loading saved model: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
        model_type = "Loaded Quantum CNN"
        
        # Check if model has quantum components
        quantum_layers = [l for l in model.layers if 'quantum' in l.name.lower()]
        if quantum_layers:
            print(f"  ✓ Model contains {len(quantum_layers)} quantum layer(s)")
        else:
            print("  ⚠ No quantum layers detected in loaded model")
            model_type = "Loaded Classical CNN"
    
    elif args.quantum_weights:
        print(f"\n🔬 Creating model with quantum weights: {args.quantum_weights}")
        model = create_model_with_quantum_weights(args.quantum_weights, 
                                                input_shape=(128, 128, 1))
        model_type = "Quantum CNN (from weights)"
        
        print("🔒 QUANTUM WEIGHTS STATUS: LOADED AND FROZEN")
        print("   ✓ Quantum parameters loaded from saved file")
        print("   ✓ No gradient updates during inference")
        print("   ✓ Completely deterministic predictions")
        
        # Load and display quantum weights metadata
        _, metadata = load_quantum_weights(args.quantum_weights)
        print(f"   Quantum configuration: {metadata.get('n_qubits', 'unknown')} qubits, "
              f"{metadata.get('n_layers', 'unknown')} layers")
        if 'final_val_accuracy' in metadata:
            print(f"   Training accuracy: {metadata['final_val_accuracy']:.4f}")
    
    # Evaluate model
    results = evaluate_model(model, test_images, test_labels, 
                           args.output_dir, model_type)
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Plots saved to: {args.output_dir}")
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Model Type: {model_type}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")
    print(f"Test F1-Score: {results['f1']:.4f}")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")


if __name__ == "__main__":
    main()