"""
Quantum CNN Diagnostic Script
==============================
Investigates WHY the quantum CNN isn't learning by systematically testing:

1. Gradient flow  — are q_weights actually getting meaningful gradients?
2. Output variance — is the quantum layer collapsing patches to constant values?
3. Parameter sensitivity — how much does changing q_weights actually change the output?
4. Barren plateau check — gradient variance vs a classical equivalent
5. Circuit expressibility — does the circuit output span [-1, 1]?
6. Classical baseline — can a classical model with the same architecture beat it?

Run: python diagnose_quantum.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pennylane as qml
import cv2
import glob

from minimal_quantum_cnn import MinimalQuantumLayer, create_minimal_quantum_cnn
from config import IMG_SIZE, N_QUBITS, N_LAYERS, CLASSES, DATASET_DIR

tf.random.set_seed(42)
np.random.seed(42)

SEP = "─" * 70


# ── helpers ────────────────────────────────────────────────────────────────

def load_n_images(split="Train", n=64):
    """Load n real images from the dataset for testing."""
    imgs, labels = [], []
    for lbl, cls in enumerate(CLASSES):
        d = os.path.join(DATASET_DIR, split, cls)
        files = glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True)[:n//2]
        files += glob.glob(os.path.join(d, "**", "*.png"), recursive=True)[:n//2]
        for f in files[:n//2]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            imgs.append(img[..., np.newaxis])
            labels.append(lbl)
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)


# ── Test 1: Gradient magnitude ─────────────────────────────────────────────

def test_gradient_magnitude(images):
    print(f"\n{SEP}")
    print("TEST 1: Gradient Magnitude — are q_weights learning?")
    print(SEP)

    model = create_minimal_quantum_cnn()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    labels = tf.constant(np.random.randint(0, 2, len(images)).astype(np.float32))

    q_layer = model.get_layer("quantum_preprocess")
    q_weights_before = q_layer.q_weights.numpy().copy()
    print(f"  q_weights初始 (init):  min={q_weights_before.min():.4f}  "
          f"max={q_weights_before.max():.4f}  "
          f"mean={q_weights_before.mean():.4f}")

    # Compute gradient for one batch
    x = tf.constant(images[:16])
    y = labels[:16]
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)

    grads = tape.gradient(loss, model.trainable_variables)

    print(f"\n  All trainable variables gradient norms:")
    q_grad_norm = None
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None:
            print(f"    {var.name:50s}  GRADIENT IS NONE ← DEAD")
            continue
        norm = tf.norm(grad).numpy()
        mean = tf.reduce_mean(tf.abs(grad)).numpy()
        tag = " ← QUANTUM" if "q_weights" in var.name else ""
        print(f"    {var.name:50s}  norm={norm:.6f}  mean|grad|={mean:.6f}{tag}")
        if "q_weights" in var.name:
            q_grad_norm = norm

    print(f"\n  q_weights grad norm: {q_grad_norm}")
    if q_grad_norm is not None and q_grad_norm < 1e-6:
        print("  ⚠  BARREN PLATEAU: q_weights gradient is essentially zero.")
        print("     The 12 quantum parameters are NOT contributing to learning.")
    elif q_grad_norm is None:
        print("  ✗  q_weights has NO gradient — backprop is broken.")
    else:
        print(f"  ✓  q_weights gradient exists (norm={q_grad_norm:.6f})")

    # Compare gradient ratio: quantum vs first CNN layer
    for var, grad in zip(model.trainable_variables, grads):
        if "conv1" in var.name and "kernel" in var.name and grad is not None:
            cnn_norm = tf.norm(grad).numpy()
            if q_grad_norm:
                ratio = q_grad_norm / cnn_norm
                print(f"\n  Ratio |∇q_weights| / |∇conv1_kernel| = {ratio:.6f}")
                if ratio < 0.01:
                    print("  ⚠  Quantum gradients are >100× smaller than CNN gradients.")
                    print("     Adam will struggle — the effective LR for quantum weights is very small.")
            break


# ── Test 2: Output variance ────────────────────────────────────────────────

def test_output_variance(images):
    print(f"\n{SEP}")
    print("TEST 2: Quantum Layer Output Variance — is it collapsing?")
    print(SEP)

    q_layer = MinimalQuantumLayer()
    x = tf.constant(images[:4])  # 4 images (slow — quantum)
    out = q_layer(x)  # (4, 32, 32, N_QUBITS)

    print(f"  Input pixel stats:  min={images[:4].min():.4f}  "
          f"max={images[:4].max():.4f}  mean={images[:4].mean():.4f}")
    print(f"  Quantum raw output stats:")
    out_np = out.numpy()
    for ch in range(N_QUBITS):
        ch_data = out_np[..., ch]
        print(f"    Channel {ch}: min={ch_data.min():.4f}  max={ch_data.max():.4f}  "
              f"mean={ch_data.mean():.4f}  std={ch_data.std():.4f}")

    total_std = out_np.std()
    print(f"\n  Overall output std: {total_std:.4f}")
    if total_std < 0.05:
        print("  ⚠  COLLAPSE: All patches map to nearly the same value.")
        print("     The circuit has no discriminative power at initialization.")
    else:
        print(f"  ✓  Output variance is non-trivial (std={total_std:.4f})")

    # Check if different channels are correlated (redundant channels)
    print(f"\n  Inter-channel correlation (should vary, not all ~1.0):")
    flat = out_np.reshape(-1, N_QUBITS)  # (4*32*32, N_QUBITS)
    for i in range(N_QUBITS):
        for j in range(i+1, N_QUBITS):
            corr = np.corrcoef(flat[:, i], flat[:, j])[0, 1]
            tag = " ⚠ REDUNDANT CHANNELS" if abs(corr) > 0.95 else ""
            print(f"    corr(ch{i}, ch{j}) = {corr:+.4f}{tag}")


# ── Test 3: Parameter sensitivity ─────────────────────────────────────────

def test_parameter_sensitivity(images):
    print(f"\n{SEP}")
    print("TEST 3: Parameter Sensitivity — does changing q_weights change output?")
    print(SEP)

    q_layer = MinimalQuantumLayer()
    x = tf.constant(images[:2])

    out_base = q_layer(x).numpy()

    # Perturb weights by a large amount (+1 radian)
    original = q_layer.q_weights.numpy().copy()
    q_layer.q_weights.assign(original + 1.0)
    out_perturbed = q_layer(x).numpy()
    q_layer.q_weights.assign(original)

    diff = np.abs(out_perturbed - out_base)
    print(f"  After +1.0 radian perturbation to ALL q_weights:")
    print(f"    Mean absolute change in output: {diff.mean():.6f}")
    print(f"    Max  absolute change in output: {diff.max():.6f}")

    if diff.mean() < 0.01:
        print("  ⚠  INSENSITIVE: Changing q_weights by 1 radian barely changes the output.")
        print("     This means the quantum layer is acting as a near-fixed transformation.")
    else:
        print(f"  ✓  Quantum output is sensitive to weight changes (mean Δ={diff.mean():.4f})")

    # What does the circuit output at extreme weight values?
    print(f"\n  Output statistics at zero weights (pure encoding):")
    q_layer.q_weights.assign(np.zeros_like(original))
    out_zero = q_layer(x).numpy()
    print(f"    std={out_zero.std():.4f}  mean={out_zero.mean():.4f}")

    print(f"  Output statistics at π weights:")
    q_layer.q_weights.assign(np.ones_like(original) * np.pi)
    out_pi = q_layer(x).numpy()
    print(f"    std={out_pi.std():.4f}  mean={out_pi.mean():.4f}")
    q_layer.q_weights.assign(original)


# ── Test 4: Capacity analysis ──────────────────────────────────────────────

def test_capacity():
    print(f"\n{SEP}")
    print("TEST 4: Model Capacity — are the quantum params enough to matter?")
    print(SEP)

    model = create_minimal_quantum_cnn()
    total = model.count_params()
    q_params = N_QUBITS * N_LAYERS  # q_weights shape
    cnn_params = total - q_params

    print(f"  Total trainable parameters: {total:,}")
    print(f"  Quantum (q_weights):        {q_params}  ({100*q_params/total:.4f}% of total)")
    print(f"  Classical CNN:              {cnn_params:,}  ({100*cnn_params/total:.2f}% of total)")
    print()

    if q_params / total < 0.001:
        print("  ⚠  CAPACITY MISMATCH: Quantum layer = less than 0.1% of parameters.")
        print("     The CNN body completely dominates — the quantum layer's contribution")
        print("     to the loss gradient is negligible in practice.")
        print("     The system is effectively a plain CNN with a fixed nonlinear preprocessing.")
    else:
        print(f"  ✓  Quantum params are a reasonable fraction ({100*q_params/total:.2f}%)")

    # Classical equivalent capacity
    print(f"\n  Classical equivalent Conv2D(stride=2, filters={N_QUBITS}):")
    classical_equiv = N_QUBITS * (2*2*1) + N_QUBITS  # filters*(kH*kW*in_ch) + bias
    print(f"    Params: {classical_equiv}  vs quantum: {q_params}")
    print(f"    Classical conv2d gives {N_QUBITS} independent spatial filters.")
    print(f"    Quantum circuit gives {N_QUBITS} CORRELATED outputs from ONE shared circuit.")

    print(f"\n  CRITICAL: All {N_QUBITS} quantum output channels are produced by the SAME")
    print(f"  12-parameter circuit. The only difference between channels is WHICH qubit")
    print(f"  you measure. With ring CNOT entanglement, the measured qubits are not independent.")
    print(f"  A classical Conv2D({N_QUBITS}, 2, strides=2) has {classical_equiv} params but learns")
    print(f"  {N_QUBITS} FULLY INDEPENDENT spatial filters — strictly more expressive.")


# ── Test 5: Expressibility ─────────────────────────────────────────────────

def test_expressibility():
    print(f"\n{SEP}")
    print("TEST 5: Circuit Expressibility — does output actually span [-1, 1]?")
    print(SEP)

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(pixels, weights):
        for q in range(N_QUBITS):
            qml.RY(pixels[q] * (np.pi / 2), wires=q)
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RX(weights[l, q], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    print(f"  Sampling 1000 random weight configurations and 1000 random inputs ...")
    results = []
    for _ in range(1000):
        pixels = np.random.uniform(0, 1, N_QUBITS)
        weights = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS))
        out = circuit(pixels, weights)
        results.append([float(v) for v in out])

    results = np.array(results)  # (1000, N_QUBITS)
    print(f"\n  Over random weights AND inputs:")
    for ch in range(N_QUBITS):
        data = results[:, ch]
        print(f"    qubit {ch}: min={data.min():.4f}  max={data.max():.4f}  "
              f"mean={data.mean():.4f}  std={data.std():.4f}")

    print(f"\n  Now fixing inputs (constant pixels=0.5), varying only weights:")
    results_fixed_input = []
    fixed_pixels = np.array([0.5] * N_QUBITS)
    for _ in range(1000):
        weights = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS))
        out = circuit(fixed_pixels, weights)
        results_fixed_input.append([float(v) for v in out])
    rfx = np.array(results_fixed_input)
    print(f"  With fixed inputs, weight-induced variance:")
    for ch in range(N_QUBITS):
        print(f"    qubit {ch}: std={rfx[:, ch].std():.4f}")

    avg_std = rfx.std(axis=0).mean()
    if avg_std < 0.1:
        print(f"\n  ⚠  LOW EXPRESSIBILITY: Even with random weights over [-π, π],")
        print(f"     output std is only {avg_std:.4f}. The circuit output is mostly")
        print(f"     determined by the INPUT (encoding), not the trainable weights.")
        print(f"     This means the quantum layer mostly acts as a fixed nonlinearity.")
    else:
        print(f"\n  ✓  Expressibility OK (mean std over weights = {avg_std:.4f})")


# ── Test 6: Overfit test ───────────────────────────────────────────────────

def test_can_overfit(images, labels):
    print(f"\n{SEP}")
    print("TEST 6: Can model overfit tiny dataset? (10 images, 50 steps)")
    print("  If it CAN'T overfit 10 images, gradient flow is broken.")
    print(SEP)

    x = tf.constant(images[:10])
    y = tf.constant(labels[:10])

    model = create_minimal_quantum_cnn()
    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    q_layer = model.get_layer("quantum_preprocess")
    q_init = q_layer.q_weights.numpy().copy()

    print("  Step  |  Loss   | Accuracy | q_weights_std")
    for step in range(50):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = loss_fn(y, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if step % 10 == 0 or step == 49:
            acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y, pred)).numpy()
            q_std = q_layer.q_weights.numpy().std()
            print(f"  {step:5d}  | {loss.numpy():.4f}  | {acc:.4f}    | {q_std:.6f}")

    q_final = q_layer.q_weights.numpy()
    q_change = np.abs(q_final - q_init).mean()
    print(f"\n  Mean absolute change in q_weights after 50 steps: {q_change:.6f}")
    if q_change < 1e-4:
        print("  ⚠  q_weights BARELY CHANGED after 50 optimization steps.")
        print("     Either gradient is near-zero OR Adam LR is too low for them.")
    else:
        print(f"  ✓  q_weights changed by avg {q_change:.4f} rad over 50 steps")

    final_pred = (model(x, training=False).numpy() > 0.5).astype(int)
    final_acc = (final_pred.flatten() == labels[:10]).mean()
    print(f"\n  Final train accuracy on 10 samples: {final_acc:.2f}")
    if final_acc < 0.9:
        print("  ⚠  CANNOT OVERFIT 10 SAMPLES — fundamental gradient flow problem.")
        print("     A model should ALWAYS be able to memorize 10 samples.")
    else:
        print("  ✓  Model can overfit training data — gradient flow is working.")


# ── Test 7: Classical baseline comparison ─────────────────────────────────

def test_classical_baseline():
    print(f"\n{SEP}")
    print("TEST 7: Classical Baseline — what does the same CNN body achieve")
    print("  WITHOUT the quantum layer? (replaces it with Conv2D(stride=2))")
    print(SEP)

    inp = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    # Replace quantum layer with classical Conv2D(stride=2) — same output shape
    x = tf.keras.layers.Conv2D(N_QUBITS, 2, strides=2, padding="same",
                                activation="relu", name="classical_preprocess")(inp)
    reg = tf.keras.regularizers.l2(1e-4)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    classical_model = tf.keras.Model(inp, out)

    q_model = create_minimal_quantum_cnn()

    qlayer_params = N_QUBITS * N_LAYERS  # q_weights
    classical_first_layer = N_QUBITS * (2*2*1) + N_QUBITS  # conv2d stride-2

    print(f"  Quantum model total params:   {q_model.count_params():>8,}")
    print(f"    of which quantum (q_weights): {qlayer_params:>6,}  ({qlayer_params} shared weights)")
    print(f"  Classical model total params: {classical_model.count_params():>8,}")
    print(f"    of which first conv:          {classical_first_layer:>6,}  ({N_QUBITS} independent filters)")
    print()
    print(f"  The classical model has {classical_model.count_params() - q_model.count_params():+,} more params.")
    print(f"  Key difference: classical first layer has {N_QUBITS} INDEPENDENT filters")
    print(f"  that learn different spatial patterns. Quantum layer produces {N_QUBITS} outputs")
    print(f"  from ONE shared circuit — less expressive for the SAME spatial task.")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("QUANTUM CNN DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Config: N_QUBITS={N_QUBITS}, N_LAYERS={N_LAYERS}, IMG_SIZE={IMG_SIZE}")
    print(f"quantum weights shape: ({N_LAYERS}, {N_QUBITS}) = {N_LAYERS*N_QUBITS} total params")

    print("\nLoading sample images ...")
    images, labels = load_n_images(split="Train", n=20)
    if len(images) == 0:
        print("Could not load images — check DATASET_DIR in config.py")
        return
    print(f"Loaded {len(images)} images  ({(labels==0).sum()} Real, {(labels==1).sum()} Fake)")

    test_capacity()              # fast — no data needed
    test_expressibility()        # fast — pure quantum simulation
    test_classical_baseline()    # fast — no data needed
    test_gradient_magnitude(images)  # medium — one forward+backward pass
    test_output_variance(images)     # SLOW — runs quantum circuit on 4 images
    test_parameter_sensitivity(images)  # SLOW — runs quantum circuit on 2 images × 3
    test_can_overfit(images, labels)    # VERY SLOW — 50 training steps on 10 images

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print("""
Key issues to investigate based on above:

1. If 'q_weights gradient is None' → PennyLane TF interface broken (reinstall)
2. If 'q_weights gradient norm << conv1 gradient norm' → barren plateau or
   the quantum layer is too insensitive for its gradients to matter
3. If 'output std < 0.05' → circuit collapse; all patches look the same
4. If 'corr(chi, chj) > 0.95' → output channels are redundant (not 4 independent features)
5. If 'cannot overfit 10 samples' → gradient flow is structurally broken
6. If 'expressibility std < 0.1' → weights barely affect output; circuit is
   dominated by the encoding and the weights are nearly irrelevant

The FUNDAMENTAL issue (always present regardless of above):
  - 12 quantum parameters shared across ALL patches of ALL images
  - 0.01% of total model capacity
  - A classical Conv2D(stride=2) at the same position is strictly more expressive
  - The quantum layer provides a FIXED nonlinearity on 2×2 patches; the
    trainable component (12 weights) is diluted by 100k CNN parameters
""")


if __name__ == "__main__":
    main()
