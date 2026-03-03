"""
QViT Deepfake Detection — CNN Evaluation
==========================================
Evaluates the trained CNN classifier on the held-out test set.

Metrics reported:
    • Accuracy, Precision, Recall, F1-score
    • AUC-ROC
    • Confusion matrix heatmap
    • ROC curve
    • Prediction probability distribution

Usage:
    python evaluate_cnn.py
    python evaluate_cnn.py --model_path ./content/cnn_output/cnn_model.keras \
                           --input_dir  ./content/preprocessed \
                           --output_dir ./content/cnn_results
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # no display needed on compute instance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve,
)

from config import PREPROCESSED_DIR, CLASSES
from cnn_model import create_cnn_classifier


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"CNN  (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — CNN", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_probability_distribution(probs, labels, save_path):
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
    ax.set_title("Prediction Distribution — CNN", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--input_dir",  default=None)
    parser.add_argument("--output_dir", default=None)
    args, _ = parser.parse_known_args()

    model_path = (args.model_path
                  or os.environ.get("AZUREML_CNN_MODEL_PATH")
                  or "./content/cnn_output/cnn_model.keras")
    input_dir  = (args.input_dir
                  or os.environ.get("AZUREML_PREPROCESSED_DIR")
                  or PREPROCESSED_DIR)
    output_dir = (args.output_dir
                  or os.environ.get("AZUREML_RESULTS_DIR")
                  or "./content/cnn_results")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("EVALUATION — CNN Deepfake Classifier")
    print("=" * 60)

    # ── Load test data ──────────────────────────
    print(f"\n📂 Loading test data from: {input_dir}")
    test_images = np.load(os.path.join(input_dir, "test_images.npy")).astype(np.float32)
    test_labels = np.load(os.path.join(input_dir, "test_labels.npy")).astype(np.int32)

    print(f"  Test: {test_images.shape}, labels: {test_labels.shape}")
    unique, counts = np.unique(test_labels, return_counts=True)
    for cls, n in zip(unique, counts):
        print(f"    {CLASSES[int(cls)]}: {n} images")

    # ── Load model ───────────────────────────────
    print(f"\n🧠 Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"create_cnn_classifier": create_cnn_classifier},
        compile=False,   # skip optimizer restore — not needed for inference
    )
    model.summary()

    # ── Predict ──────────────────────────────────
    print("\n🔍 Running inference on test set...")
    probs = model.predict(test_images, batch_size=32, verbose=1).flatten()
    preds = (probs >= 0.5).astype(np.int32)

    # ── Metrics ──────────────────────────────────
    acc       = accuracy_score(test_labels, preds)
    auc       = roc_auc_score(test_labels, probs)
    f1        = f1_score(test_labels, preds)
    precision = precision_score(test_labels, preds)
    recall    = recall_score(test_labels, preds)
    cm        = confusion_matrix(test_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(test_labels, probs)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred:real  Pred:fake")
    print(f"  Actual:real    {tn:5d}      {fp:5d}")
    print(f"  Actual:fake    {fn:5d}      {tp:5d}")
    print(f"\n  Per-class report:")
    print(classification_report(test_labels, preds, target_names=[CLASSES[0], CLASSES[1]]))

    # ── Save graphs ──────────────────────────────
    print("\n📊 Saving graphs...")
    plot_confusion_matrix(
        cm, list(CLASSES),
        os.path.join(output_dir, "cnn_confusion_matrix.png"),
    )
    plot_roc_curve(
        fpr, tpr, auc,
        os.path.join(output_dir, "cnn_roc_curve.png"),
    )
    plot_probability_distribution(
        probs, test_labels,
        os.path.join(output_dir, "cnn_prob_distribution.png"),
    )

    # ── Save results JSON ─────────────────────────
    results = {
        "accuracy":  float(acc),
        "auc":       float(auc),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }
    json_path = os.path.join(output_dir, "cnn_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {json_path}")

    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
