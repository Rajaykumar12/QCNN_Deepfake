"""
QViT Deepfake Detection — CNN Baseline Training
================================================
Trains the CNN classifier directly on preprocessed dual-channel
images (128×128×2). Reads from PREPROCESSED_DIR — no quantum
convolution step needed.

Usage:
    python train_cnn.py
    python train_cnn.py --input_dir /path/to/preprocessed --output_dir /path/to/output
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import math
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import warnings
import logging

# Suppress harmless TensorFlow casting warnings from PennyLane's complex→float conversion
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging verbosity
warnings.filterwarnings('ignore', category=UserWarning, message='.*casting an input of type complex128.*')

# Suppress TensorFlow logger from printing the casting warning
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)

# Try to import cv2, fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠ OpenCV not available - some fallback features may not work")

from config import (
    TRAIN_DIR, PREPROCESSED_DIR, BATCH_SIZE, EPOCHS,
    WEIGHT_DECAY, WARMUP_EPOCHS, EARLY_STOPPING_PATIENCE,
    VAL_SPLIT, RANDOM_SEED, CLASSES,
)

# Import modules with explicit path handling for Azure ML
# Add fallback for quantum imports to handle Python 3.8 compatibility
quantum_available = True
try:
    from cnn_model import create_quantum_cnn_classifier
    from quantum_preprocessing import create_raw_image_loader, auto_detect_dataset_structure
    from quantum_weights_utils import save_quantum_weights
except ImportError as e:
    print(f"⚠ Quantum imports failed (trying baseline_cnn prefix): {e}")
    try:
        # Try with baseline_cnn prefix for Azure ML environment
        from baseline_cnn.cnn_model import create_quantum_cnn_classifier
        from baseline_cnn.quantum_preprocessing import create_raw_image_loader, auto_detect_dataset_structure  
        from baseline_cnn.quantum_weights_utils import save_quantum_weights
    except ImportError as e2:
        print(f"❌ Quantum preprocessing unavailable (Python 3.8?): {e2}")
        print(f"🔄 Falling back to classical mode only...")
        quantum_available = False
        
        # Create fallback functions for classical mode
        def create_raw_image_loader(data_dir, classes=['fake', 'real'], img_size=128, 
                                  max_total_samples=None, max_samples_per_class=None,
                                  balanced_sampling=True):
            import cv2
            from sklearn.utils import shuffle
            
            print(f"📁 Loading images from: {data_dir}")
            images = []
            labels = []
            
            for class_idx, class_name in enumerate(classes):
                class_path = os.path.join(data_dir, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                class_images = []
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            # Load and resize image
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (img_size, img_size))
                            img = img.astype(np.float32) / 255.0
                            class_images.append(img)
                        except Exception as e:
                            continue
                
                # Apply sampling limits
                if max_samples_per_class and len(class_images) > max_samples_per_class:
                    class_images = class_images[:max_samples_per_class]
                
                # Add to main lists
                images.extend(class_images) 
                labels.extend([class_idx] * len(class_images))
                print(f"  {class_name}: {len(class_images)} images")
            
            # Convert to numpy arrays
            images = np.array(images)
            labels = np.array(labels)
            
            # Add channel dimension (grayscale)
            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=-1)
            
            # Shuffle data
            images, labels = shuffle(images, labels, random_state=42)
            
            # Apply total sample limit
            if max_total_samples and len(images) > max_total_samples:
                images = images[:max_total_samples]
                labels = labels[:max_total_samples]
            
            print(f"✓ Loaded {len(images)} total images")
            return images, labels
            
        def auto_detect_dataset_structure(dataset_dir):
            # Simple fallback structure detection
            train_dir = os.path.join(dataset_dir, 'Train')
            test_dir = os.path.join(dataset_dir, 'Test')
            
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                return {
                    'has_train_test_structure': True,
                    'train_dir': train_dir,
                    'test_dir': test_dir,
                    'data_dirs': []
                }
            else:
                # Find subdirectories with images
                data_dirs = []
                for item in os.listdir(dataset_dir):
                    item_path = os.path.join(dataset_dir, item)
                    if os.path.isdir(item_path):
                        data_dirs.append(item_path)
                
                return {
                    'has_train_test_structure': False,
                    'train_dir': None,
                    'test_dir': None,
                    'data_dirs': data_dirs
                }
        
        def create_quantum_cnn_classifier(input_shape, use_quantum_preprocessing=False):
            # Import here to avoid circular imports
            from tensorflow.keras import layers, models
            
            print("🔧 Creating classical CNN (quantum unavailable)...")
            
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.Conv2D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.Conv2D(128, 3, activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            
            return model
        
        def save_quantum_weights(*args, **kwargs):
            print("⚠ Quantum weights not available in classical fallback mode")
            pass


# ──────────────────────────────────────────────
# Learning Rate Schedule  (same as ViT trainer)
# ──────────────────────────────────────────────

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, initial_lr, warmup_steps, decay_steps, min_lr=1e-6):
        super().__init__()
        self.initial_lr  = float(initial_lr)
        self.warmup_steps = float(warmup_steps)
        self.decay_steps  = float(decay_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_pct = tf.minimum(step / tf.maximum(self.warmup_steps, 1.0), 1.0)
        warmup_lr  = self.initial_lr * warmup_pct

        decay_step = tf.maximum(step - self.warmup_steps, 0.0)
        decay_pct  = tf.minimum(decay_step / tf.maximum(self.decay_steps, 1.0), 1.0)
        cosine_lr  = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * decay_pct)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":   self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps":  self.decay_steps,
            "min_lr":       self.min_lr,
        }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── Enable mixed precision for faster training ───
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✓ Mixed precision training enabled")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None,
                        help="Path to training images directory")
    parser.add_argument("--test_dir", default=None, 
                        help="Path to test images directory (optional)")
    parser.add_argument("--dataset_dir", default=None,
                        help="Path to dataset root (auto-detects Train/Test structure)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save model + history")
    parser.add_argument("--use_quantum", action="store_true", default=True,
                        help="Use quantum preprocessing (default: True)")
    parser.add_argument("--classical_mode", action="store_true", default=False,
                        help="Use classical preprocessing instead of quantum")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum TOTAL dataset size including train+val+test (default: 10000)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Maximum samples per class (overrides max_samples)")
    parser.add_argument("--no_balanced_sampling", action="store_true",
                        help="Disable balanced sampling across classes")
    args, _ = parser.parse_known_args()
    
    # Determine preprocessing mode (disable quantum if not available)
    use_quantum_preprocessing = args.use_quantum and not args.classical_mode and quantum_available
    
    if args.use_quantum and not quantum_available:
        print("⚠ Quantum preprocessing requested but unavailable - using classical mode")
    
    # Prepare output directory
    output_dir = (
        args.output_dir
        or os.environ.get("AZUREML_MODEL_DIR")
        or "./content"
    )

    model_save_path = os.path.join(output_dir, "quantum_cnn_model.keras" if use_quantum_preprocessing else "cnn_model.keras")
    history_save_path = os.path.join(output_dir, "quantum_cnn_training_history.npy" if use_quantum_preprocessing else "cnn_training_history.npy")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING — {'Quantum-Enhanced' if use_quantum_preprocessing else 'Classical'} CNN Deepfake Classifier")
    print("=" * 60)
    print(f"Mode: {'🔬 Quantum Preprocessing' if use_quantum_preprocessing else '🔧 Classical Preprocessing'}")
    if use_quantum_preprocessing:
        print("🔬 QUANTUM PARAMETER BEHAVIOR:")
        print("   ✓ Training Phase: Quantum weights are LEARNABLE (updated via backprop)")
        print("   ✓ Validation Phase: Quantum weights are FROZEN (no updates)")
        print("   ✓ Testing Phase: Quantum weights are LOADED from saved file (completely frozen)")
        print("   ✓ End-to-End Training: Quantum + Classical parameters optimized together")

    # ── Flexible Dataset Loading ──────────────────────────────────
    print(f"\n📂 Preparing dataset...")

    # Allow CLI override before searching defaults
    dataset_dir = args.dataset_dir
    if dataset_dir:
        # resolve relative paths so validation is reliable
        dataset_dir = os.path.abspath(dataset_dir)
        print(f"👀 Using dataset path from argument: {dataset_dir}")
        if not os.path.exists(dataset_dir):
            print(f"❌ Provided dataset_dir does not exist: {dataset_dir}")
            dataset_dir = None

    # Hardcoded dataset paths (prioritize Azure ML compute instance)
    dataset_paths = [
        '/home/azureuser/cloudfiles/code/deepfake_data/Dataset',
        './deepfake_data/Dataset',
        '/mnt/batch/tasks/shared/LS_root/mounts/clusters/rajaykumar129591/cloudfiles/code/deepfake_data/Dataset'
    ]

    if not dataset_dir:
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_dir = path
                print(f"✓ Found dataset at: {dataset_dir}")
                break
    
    if not dataset_dir:
        print("❌ Dataset not found at any expected location!")
        print("Expected paths:")
        if args.dataset_dir:
            print(f"  - provided: {args.dataset_dir}")
        for path in dataset_paths:
            print(f"  - {path}")
        
        # Create synthetic test dataset for development/testing
        print("🔧 Creating synthetic test dataset...")
        test_dataset_dir = './synthetic_test_dataset'
        os.makedirs(test_dataset_dir, exist_ok=True)
        
        # Create Train and Test directories
        for split in ['Train', 'Test']:
            for class_name in ['real', 'fake']:
                class_dir = os.path.join(test_dataset_dir, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Generate synthetic images (simple patterns)
                n_images = 50 if split == 'Train' else 20
                for i in range(n_images):
                    # Create synthetic image based on class
                    if class_name == 'real':
                        # Real: smooth gradient
                        img = np.random.rand(128, 128) * 0.5 + 0.3
                    else:
                        # Fake: noisy pattern  
                        img = np.random.rand(128, 128) * 0.8 + 0.1
                    
                    # Save synthetic image using PIL
                    from PIL import Image
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                    img_path = os.path.join(class_dir, f'synthetic_{i:03d}.png')
                    img_pil.save(img_path)
        
        dataset_dir = test_dataset_dir
        print(f"✓ Created synthetic test dataset at: {dataset_dir}")
        print("⚠ Using synthetic data for testing - replace with real dataset for actual training")
    
    # Auto-detect structure from dataset root
    structure_info = auto_detect_dataset_structure(dataset_dir)
    print(f"Auto-detected dataset structure: {structure_info}")

    # Compatibility layer for older versions of auto_detect_dataset_structure
    # previous implementation returned 'has_train_test_structure'; new version
    # uses 'has_train_subdir'/'has_test_subdir' and 'suggested_mode'.
    if 'has_train_test_structure' not in structure_info:
        # derive old-style flags
        structure_info['has_train_test_structure'] = (
            structure_info.get('has_train_subdir', False) and
            structure_info.get('has_test_subdir', False)
        )
        # ensure train_dir/test_dir exist (new format already sets them)
        structure_info.setdefault('train_dir', None)
        structure_info.setdefault('test_dir', None)
        # add data_dirs field to avoid KeyError later
        structure_info.setdefault('data_dirs', [])

    if structure_info['has_train_test_structure']:
        # Found Train/Test directories - need to coordinate limits to stay within total
        train_dir = structure_info['train_dir']
        test_dir = structure_info['test_dir']
        
        print(f"Using separate Train/Test structure:")
        print(f"  Train directory: {train_dir}")
        print(f"  Test directory: {test_dir}")
        print(f"  Total dataset limit: {args.max_samples} images (train+test combined)")
        
        # Calculate split for train vs test to stay within total limit
        train_ratio = 0.9  # 90% for training, 10% for testing
        max_train_samples = int(args.max_samples * train_ratio)
        max_test_samples = args.max_samples - max_train_samples
        
        print(f"  Allocating: {max_train_samples} train + {max_test_samples} test = {args.max_samples} total")
        
        # Load train data with allocated limit (loader may return 2 or more outputs)
        loader_output = create_raw_image_loader(
            train_dir, classes=CLASSES, img_size=128,
            max_total_samples=max_train_samples,
            max_samples_per_class=args.max_per_class,
            balanced_sampling=not args.no_balanced_sampling
        )
        # handle varying return formats
        if isinstance(loader_output, tuple) and len(loader_output) >= 2:
            all_images, all_labels = loader_output[0], loader_output[1]
            # if loader already returned validation split, use it to avoid re-splitting
            if len(loader_output) == 4:
                train_images, train_labels, val_images, val_labels = loader_output
                print("ℹ Using validation set provided by loader (train_val mode)")
            else:
                # fall back to manual split
                from sklearn.model_selection import train_test_split
                train_images, val_images, train_labels, val_labels = train_test_split(
                    all_images, all_labels, test_size=args.val_split,
                    random_state=RANDOM_SEED, stratify=all_labels
                )
        else:
            raise ValueError(f"Unexpected return value from create_raw_image_loader: {loader_output}")
        
        # Load test data with remaining allocation
        loader_output = create_raw_image_loader(
            test_dir, classes=CLASSES, img_size=128,
            max_total_samples=max_test_samples,
            max_samples_per_class=None,  # Use total limit instead
            balanced_sampling=not args.no_balanced_sampling
        )
        if isinstance(loader_output, tuple) and len(loader_output) >= 2:
            test_images, test_labels = loader_output[0], loader_output[1]
            if len(loader_output) > 2:
                print("⚠ Loader returned extra outputs for test set; ignoring them")
        else:
            raise ValueError(f"Unexpected return value from create_raw_image_loader: {loader_output}")
        
        total_dataset_size = len(all_images) + len(test_images)
        print(f"\n📊 Final dataset allocation:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val:   {len(val_images)} images")
        print(f"  Test:  {len(test_images)} images")
        print(f"  Total: {total_dataset_size} images (limit: {args.max_samples})")
        has_test_set = True
        
    else:
        # Single directory structure - split for train/val
        data_dir = structure_info['data_dirs'][0] if structure_info['data_dirs'] else dataset_dir
        print(f"Using single directory with train/val split: {data_dir}")
        
        loader_output = create_raw_image_loader(
            data_dir, classes=CLASSES, img_size=128,
            max_total_samples=args.max_samples,
            max_samples_per_class=args.max_per_class,
            balanced_sampling=not args.no_balanced_sampling
        )
        if isinstance(loader_output, tuple) and len(loader_output) >= 2:
            all_images, all_labels = loader_output[0], loader_output[1]
            if len(loader_output) == 4:
                train_images, train_labels, val_images, val_labels = loader_output
                print("ℹ Using validation set provided by loader (train_val mode)")
            else:
                from sklearn.model_selection import train_test_split
                train_images, val_images, train_labels, val_labels = train_test_split(
                    all_images, all_labels, test_size=args.val_split,
                    random_state=RANDOM_SEED, stratify=all_labels
                )
        else:
            raise ValueError(f"Unexpected return value from create_raw_image_loader: {loader_output}")
        
        # Create train/val split (no separate test set)
        from sklearn.model_selection import train_test_split
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, all_labels, test_size=args.val_split, 
            random_state=RANDOM_SEED, stratify=all_labels
        )
        has_test_set = False
    
    # Convert to float32
    train_images = train_images.astype(np.float32)
    val_images = val_images.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    val_labels = val_labels.astype(np.float32)
    
    if has_test_set:
        test_images = test_images.astype(np.float32)
        test_labels = test_labels.astype(np.float32)

    print(f"  Total images loaded: {len(all_images)}")
    print(f"  Train: {train_images.shape}, labels: {train_labels.shape}")
    print(f"  Val:   {val_images.shape},   labels: {val_labels.shape}")
    if has_test_set:
        print(f"  Test:  {test_images.shape},  labels: {test_labels.shape}")
    print(f"  Input shape: {train_images.shape[1:]}")
    print(f"  Classes: {CLASSES}")

    # ── Class weights ────────────────────────────
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    class_weight = {
        int(cls): total / (len(unique) * count)
        for cls, count in zip(unique, counts)
    }
    print(f"  Class weights: {class_weight}")

    # ── Build quantum-enhanced model ─────────────────────
    print(f"\n🧠 Building {'quantum-enhanced' if use_quantum_preprocessing else 'classical'} CNN classifier...")
    model = create_quantum_cnn_classifier(
        input_shape=train_images.shape[1:],
        use_quantum_preprocessing=use_quantum_preprocessing
    )
    model.summary()
    
    # Count quantum vs classical parameters
    quantum_weights = [w for w in model.trainable_weights if 'quantum' in w.name.lower()]
    classical_weights = [w for w in model.trainable_weights if 'quantum' not in w.name.lower()]
    
    print(f"\n📊 Model Statistics:")
    print(f"  Quantum parameters: {len(quantum_weights)}")
    print(f"  Classical parameters: {len(classical_weights)}")
    print(f"  Total trainable weights: {len(model.trainable_weights)}")
    
    # ── Simple data augmentation for raw images ──────────
    if use_quantum_preprocessing:
        print("\n🔄 Setting up raw image augmentation...")
        # Create augmentation in float32 mode (not affected by mixed precision policy)
        # Temporarily disable mixed precision for augmentation layer creation
        original_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy('float32')
        
        augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.05, fill_mode='constant', fill_value=0.0),
            layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=0.0),
            layers.RandomBrightness(0.1, value_range=[0, 1]),
            layers.RandomContrast(0.1),
        ], name='raw_image_augmentation')
        
        # Re-enable mixed precision for training
        tf.keras.mixed_precision.set_global_policy(original_policy)
        
        # Create datasets with augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    else:
        # Classical mode - standard dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset (no augmentation)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # ── LR schedule ──────────────────────────────
    steps_per_epoch = math.ceil(len(train_images) / BATCH_SIZE)
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
    decay_steps     = (EPOCHS - WARMUP_EPOCHS) * steps_per_epoch

    lr_schedule = WarmupCosineSchedule(
        initial_lr=1e-4,  # Use config value
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr=1e-7,  # Lower minimum LR
    )

    # ── Focal Loss for handling class imbalance ──
    def focal_loss(gamma=2.0, alpha=0.75):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal loss
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_factor = alpha_t * tf.pow((1 - p_t), gamma)
            
            # Binary crossentropy
            bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
            
            return focal_factor * bce
        return focal_loss_fixed
    
    # ── Compile with enhanced metrics ─────────────
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,  # Gradient clipping for stability
    )
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.75),  # Focus on hard examples
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    # ── Enhanced callbacks ───────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"),
            append=False
        ),
    ]

    # ── Train ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")
    print(f"  Mode:          {'🔬 Quantum-Enhanced' if use_quantum_preprocessing else '🔧 Classical'}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Epochs:        {EPOCHS}  (early stop patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Steps/epoch:   {steps_per_epoch}")
    print(f"  Initial LR:    1e-4")
    print(f"  Weight decay:  {WEIGHT_DECAY}")
    if use_quantum_preprocessing:
        print(f"  Quantum params: {len(quantum_weights)}")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save history ─────────────────────────────
    history_dict = {
        key: [float(v) for v in vals]
        for key, vals in history.history.items()
    }
    np.save(history_save_path, history_dict)
    print(f"\n✓ Training history saved to: {history_save_path}")
    
    # ── Save quantum weights separately ──────────
    if use_quantum_preprocessing and len(quantum_weights) > 0:
        quantum_weights_path = os.path.join(output_dir, "quantum_weights.npz")
        
        # Use the utility function to save quantum weights
        training_metadata = {
            'epoch_trained': len(history.history['loss']),
            'final_val_accuracy': float(val_acc),
            'final_val_auc': float(val_auc),
            'training_mode': 'quantum_enhanced'
        }
        
        save_quantum_weights(model, quantum_weights_path, metadata=training_metadata)
        
        # Also save a human-readable summary
        quantum_summary_path = os.path.join(output_dir, "quantum_weights_summary.txt")
        with open(quantum_summary_path, 'w') as f:
            f.write("Quantum Weights Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training completed after {len(history.history['loss'])} epochs\n")
            f.write(f"Final validation accuracy: {val_acc:.4f}\n")
            f.write(f"Final validation AUC: {val_auc:.4f}\n")
            f.write(f"Number of quantum parameters: {len(quantum_weights)}\n")
            
            try:
                quantum_layer = model.get_layer('quantum_preprocessing')
                f.write(f"Quantum circuit: {quantum_layer.n_qubits} qubits, {quantum_layer.n_layers} layers\n\n")
            except:
                f.write("Quantum circuit: settings not available\n\n")
            
            f.write("Quantum Weight Statistics:\n")
            f.write("-" * 25 + "\n")
            for i, weight in enumerate(quantum_weights):
                f.write(f"Weight {i}: {weight.name}\n")
                f.write(f"  Shape: {weight.shape}\n")
                f.write(f"  Mean: {float(tf.reduce_mean(weight)):.6f}\n")
                f.write(f"  Std: {float(tf.math.reduce_std(weight)):.6f}\n")
                f.write(f"  Min: {float(tf.reduce_min(weight)):.6f}\n")
                f.write(f"  Max: {float(tf.reduce_max(weight)):.6f}\n\n")
        
        print(f"✓ Quantum weights summary saved to: {quantum_summary_path}")

    # ── Final evaluation ─────────────────────────
    # Always evaluate on validation set
    val_results = model.evaluate(val_images, val_labels, verbose=0)
    val_loss, val_acc, val_auc, val_precision, val_recall = val_results
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)
    
    print(f"\n{'='*50}")
    print("FINAL VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"AUC:       {val_auc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"Loss:      {val_loss:.4f}")
    
    # Evaluate on test set if available
    if has_test_set:
        test_results = model.evaluate(test_images, test_labels, verbose=0)
        test_loss, test_acc, test_auc, test_precision, test_recall = test_results
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-10)
        
        print(f"\n{'='*50}")
        print("FINAL TEST RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy:  {test_acc:.4f}")
        print(f"AUC:       {test_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-Score:  {test_f1:.4f}")
        print(f"Loss:      {test_loss:.4f}")
    
    print(f"\n✓ Best model saved to: {model_save_path}")
    
    # Save final metrics with mode info
    final_metrics = {
        'mode': 'quantum' if use_quantum_preprocessing else 'classical',
        'quantum_parameters': len(quantum_weights) if use_quantum_preprocessing else 0,
        'classical_parameters': len(classical_weights),
        'has_test_set': has_test_set,
        'validation_metrics': {
            'accuracy': float(val_acc),
            'auc': float(val_auc),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1),
            'loss': float(val_loss)
        }
    }
    
    # Add test metrics if available
    if has_test_set:
        final_metrics['test_metrics'] = {
            'accuracy': float(test_acc),
            'auc': float(test_auc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'loss': float(test_loss)
        }
    
    import json
    metrics_filename = 'quantum_final_metrics.json' if use_quantum_preprocessing else 'final_metrics.json'
    with open(os.path.join(output_dir, metrics_filename), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✓ Final metrics saved to: {os.path.join(output_dir, metrics_filename)}")


if __name__ == "__main__":
    main()
