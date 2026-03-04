"""
Quantum Image Preprocessing Layer
=================================
PennyLane-based quantum preprocessing for raw images, integrated directly 
into the CNN training pipeline for end-to-end backpropagation.

This replaces traditional DCT preprocessing with trainable quantum circuits
that can learn optimal frequency-domain transformations for deepfake detection.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers

from config import IMG_SIZE, N_QUBITS, N_LAYERS


class QuantumImagePreprocessor(tf.keras.layers.Layer):
    """
    Quantum preprocessing layer that transforms raw grayscale images
    into quantum-enhanced feature representations.
    
    Input:  (batch, height, width, 1)  - Raw grayscale images [0, 1]
    Output: (batch, height//2, width//2, n_qubits)  - Quantum features
    
    The layer:
    1. Takes raw grayscale images
    2. Extracts 2x2 patches 
    3. Processes each patch through a trainable quantum circuit
    4. Outputs quantum feature maps for the CNN
    """
    
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qlayer = self._make_quantum_layer()
        
    def _make_quantum_layer(self):
        """Create the quantum circuit layer for image preprocessing."""
        
        @qml.qnode(self.dev, interface="tf", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            # Encode 2x2 image patch into quantum state
            # inputs shape: (4,) for a 2x2 patch
            for i in range(self.n_qubits):
                # Amplitude encoding with rotation gates
                qml.RX(inputs[i] * np.pi, wires=i)
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Trainable quantum layers for feature extraction
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # Define weight shapes for the quantum circuit
        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=self.n_qubits
            )
        }
        
        return qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=self.n_qubits)
    
    def call(self, x, training=None):
        """
        Process raw images through quantum preprocessing.
        
        Args:
            x: (batch, height, width, 1) - Raw grayscale images
            training: Boolean indicating training mode. In training mode,
                     quantum parameters are updatable via backprop.
                     In evaluation mode, quantum weights are frozen.
            
        Returns:
            (batch, height//2, width//2, n_qubits) - Quantum features
        """
        batch_size = tf.shape(x)[0]
        height, width = x.shape[1], x.shape[2]
        out_height, out_width = height // 2, width // 2
        
        # Remove channel dimension: (batch, height, width)
        x = tf.squeeze(x, axis=-1)
        
        # Extract 2x2 patches
        # Reshape to (batch, out_height, 2, out_width, 2)
        x = tf.reshape(x, (batch_size, out_height, 2, out_width, 2))
        # Transpose to (batch, out_height, out_width, 2, 2)
        x = tf.transpose(x, (0, 1, 3, 2, 4))
        # Flatten patches to (batch, out_height, out_width, 4)
        x = tf.reshape(x, (batch_size, out_height, out_width, 4))
        
        # Process all patches through quantum circuit
        # Flatten to (batch * out_height * out_width, 4)
        patches_flat = tf.reshape(x, (-1, 4))
        
        # Apply quantum preprocessing
        # Note: During training=True, quantum weights are trainable via backprop
        # During training=False (validation/testing), quantum weights are frozen
        quantum_features = self.qlayer(patches_flat)
        
        # Reshape back to spatial format
        # (batch, out_height, out_width, n_qubits)
        output = tf.reshape(quantum_features, 
                          (batch_size, out_height, out_width, self.n_qubits))
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config


class DualQuantumPreprocessor(tf.keras.layers.Layer):
    """
    Dual-path quantum preprocessing that creates two different quantum
    representations of the input image, similar to grayscale + DCT channels
    but using trainable quantum circuits.
    
    Input:  (batch, height, width, 1)  - Raw grayscale images
    Output: (batch, height//2, width//2, n_qubits * 2)  - Dual quantum features
    """
    
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create two independent quantum preprocessors
        # Each will learn different representations
        self.quantum_path_1 = QuantumImagePreprocessor(
            n_qubits, n_layers, name="quantum_spatial"
        )
        self.quantum_path_2 = QuantumImagePreprocessor(
            n_qubits, n_layers, name="quantum_frequency"
        )
    
    def call(self, x, training=None):
        """
        Process input through dual quantum paths.
        
        QUANTUM PARAMETER BEHAVIOR:
        - Training mode (training=True): All quantum parameters are learnable
          and updated via gradient descent with classical parameters
        - Evaluation mode (training=False): Quantum weights are frozen,
          providing consistent inference behavior
        
        Args:
            x: (batch, height, width, 1) - Raw grayscale images
            training: Boolean for training vs evaluation mode
            
        Returns:
            (batch, height//2, width//2, n_qubits * 2) - Dual quantum features
        """
        # Process through both quantum paths
        # During training=True: quantum parameters are updated via backprop
        # During training=False: quantum weights are frozen for consistent inference
        q_features_1 = self.quantum_path_1(x, training=training)  # Spatial-like quantum features
        q_features_2 = self.quantum_path_2(x, training=training)  # Frequency-like quantum features
        
        # Concatenate along channel dimension
        return tf.concat([q_features_1, q_features_2], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config


def create_raw_image_loader(image_dir, classes=("original", "fake"), img_size=IMG_SIZE, 
                           split_mode="train_val", test_dir=None, val_split=0.15, random_state=42,
                           max_total_samples=None, max_samples_per_class=None, balanced_sampling=True):
    """
    Load raw images for quantum preprocessing with flexible dataset structures.
    
    Args:
        image_dir: Base directory containing images
        classes: Tuple of class names (default: ("original", "fake"))
        img_size: Target image size
        split_mode: "train_val", "train_only", "test_only", "train_test"
        test_dir: Optional separate test directory
        val_split: Validation split ratio (used if split_mode="train_val")
        random_state: Random seed for reproducible splits
        max_total_samples: Maximum total dataset size (e.g., 10000)
        max_samples_per_class: Maximum samples per class (overrides total if specified)
        balanced_sampling: Ensure equal samples per class when limiting
        
    Returns:
        Depending on split_mode:
        - "train_val": (train_images, train_labels, val_images, val_labels)
        - "train_only": (images, labels)
        - "test_only": (images, labels)  
        - "train_test": (train_images, train_labels, test_images, test_labels)
    """
    import os
    from PIL import Image
    from sklearn.model_selection import train_test_split
    
    # Apply config defaults if not specified
    if max_total_samples is None:
        from config import TOTAL_MAX_SAMPLES
        max_total_samples = TOTAL_MAX_SAMPLES
        
    if max_samples_per_class is None:
        from config import MAX_SAMPLES_PER_CLASS
        max_samples_per_class = MAX_SAMPLES_PER_CLASS
        
    if balanced_sampling is None:
        from config import USE_BALANCED_SAMPLING
        balanced_sampling = USE_BALANCED_SAMPLING
    
    def load_images_from_dir(base_dir, classes, max_total=None, max_per_class=None, balanced=True):
        """Load images from a directory with class subdirectories and apply size limits."""
        images = []
        labels = []
        
        # Calculate per-class limits
        if max_per_class is not None:
            samples_per_class = max_per_class
        elif max_total is not None and balanced:
            samples_per_class = max_total // len(classes)
        else:
            samples_per_class = None
        
        print(f"Dataset limiting: max_total={max_total}, per_class={samples_per_class}, balanced={balanced}")
        
        all_class_images = {}  # Store images by class for balanced sampling
        
        for label_idx, class_name in enumerate(classes):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠ Directory not found: {class_dir}")
                continue
                
            class_images = []
            class_labels = []
            
            # Collect all images for this class
            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
                    if ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
                        img_path = os.path.join(root, fname)
                        try:
                            # Load and preprocess image
                            img = Image.open(img_path).convert("L")  # Grayscale
                            img = img.resize((img_size, img_size), Image.LANCZOS)
                            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
                            
                            class_images.append(img_array)
                            class_labels.append(label_idx)
                        except Exception as e:
                            print(f"⚠ Failed to load {img_path}: {e}")
            
            print(f"  {class_name}: {len(class_images)} images found")
            
            # Apply per-class limit with random sampling
            if samples_per_class is not None and len(class_images) > samples_per_class:
                import random
                random.seed(random_state)
                indices = random.sample(range(len(class_images)), samples_per_class)
                class_images = [class_images[i] for i in indices]
                class_labels = [class_labels[i] for i in indices]
                print(f"    → Limited to {len(class_images)} images")
            
            all_class_images[class_name] = (class_images, class_labels)
            images.extend(class_images)
            labels.extend(class_labels)
        
        # Apply total limit if specified and not using per-class limits
        if max_total is not None and len(images) > max_total and samples_per_class is None:
            print(f"Applying total limit: {len(images)} → {max_total} images")
            
            if balanced:
                # Balanced sampling across classes
                import random
                random.seed(random_state)
                
                target_per_class = max_total // len(classes)
                remaining = max_total % len(classes)
                
                final_images = []
                final_labels = []
                
                for idx, (class_name, (class_imgs, class_lbls)) in enumerate(all_class_images.items()):
                    class_limit = target_per_class + (1 if idx < remaining else 0)
                    if len(class_imgs) > class_limit:
                        indices = random.sample(range(len(class_imgs)), class_limit)
                        class_imgs = [class_imgs[i] for i in indices]
                        class_lbls = [class_lbls[i] for i in indices]
                    
                    final_images.extend(class_imgs)
                    final_labels.extend(class_lbls)
                    print(f"    {class_name}: {len(class_imgs)} images selected")
                
                images = final_images
                labels = final_labels
            else:
                # Random sampling across all images
                import random
                random.seed(random_state)
                indices = random.sample(range(len(images)), max_total)
                images = [images[i] for i in indices]
                labels = [labels[i] for i in indices]
        
        if len(images) == 0:
            raise ValueError(f"No images found in {base_dir}")
        
        # Convert to numpy arrays and add channel dimension
        images = np.array(images, dtype=np.float32)[..., np.newaxis]  # Add channel dim
        labels = np.array(labels, dtype=np.float32)
        
        return images, labels
    
    print(f"Loading images from: {image_dir}")
    print(f"Classes: {classes}")
    print(f"Split mode: {split_mode}")
    
    if split_mode == "train_val":
        # Load all images and split into train/val
        images, labels = load_images_from_dir(image_dir, classes, max_total_samples, max_samples_per_class, balanced_sampling)
        
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=val_split, 
            random_state=random_state, stratify=labels
        )
        
        print(f"Final dataset: Total={len(images)}, Train={len(train_images)}, Val={len(val_images)}")
        
        return train_images, train_labels, val_images, val_labels
    
    elif split_mode == "train_only":
        # Load training images only
        images, labels = load_images_from_dir(image_dir, classes, max_total_samples, max_samples_per_class, balanced_sampling)
        print(f"Training images: {len(images)}")
        return images, labels
    
    elif split_mode == "test_only":
        # Load test images only (from image_dir or test_dir)
        test_directory = test_dir if test_dir else image_dir
        images, labels = load_images_from_dir(test_directory, classes, max_total_samples, max_samples_per_class, balanced_sampling)
        print(f"Test images: {len(images)}")
        return images, labels
    
    elif split_mode == "train_test":
        # Load separate train and test sets
        if not test_dir:
            raise ValueError("test_dir must be provided when split_mode='train_test'")
        
        train_images, train_labels = load_images_from_dir(image_dir, classes, max_total_samples, max_samples_per_class, balanced_sampling)
        test_images, test_labels = load_images_from_dir(test_dir, classes, max_total_samples, max_samples_per_class, balanced_sampling) 
        
        print(f"Train images: {len(train_images)}")
        print(f"Test images: {len(test_images)}")
        
        return train_images, train_labels, test_images, test_labels
    
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}. Use 'train_val', 'train_only', 'test_only', or 'train_test'")


def auto_detect_dataset_structure(base_directory):
    """
    Automatically detect the dataset structure and suggest appropriate loading parameters.
    
    Args:
        base_directory: Path to the dataset root directory
        
    Returns:
        dict: Dictionary containing detected structure and suggested parameters
    """
    import os
    
    structure_info = {
        'base_dir': base_directory,
        'has_train_subdir': False,
        'has_test_subdir': False,
        'has_direct_classes': False,
        'suggested_mode': None,
        'train_dir': None,
        'test_dir': None,
        'classes_found': []
    }
    
    if not os.path.exists(base_directory):
        structure_info['error'] = f"Directory not found: {base_directory}"
        return structure_info
    
    # Check for common class names directly in base directory
    direct_classes = []
    for class_name in ['original', 'fake', 'real', 'authentic', 'deepfake', 'synthetic']:
        class_path = os.path.join(base_directory, class_name)
        if os.path.isdir(class_path):
            direct_classes.append(class_name)
    
    # Check for Train/Test subdirectories
    train_candidates = ['Train', 'train', 'training', 'TRAIN']
    test_candidates = ['Test', 'test', 'testing', 'TEST']
    
    train_dir = None
    test_dir = None
    
    for candidate in train_candidates:
        candidate_path = os.path.join(base_directory, candidate)
        if os.path.isdir(candidate_path):
            train_dir = candidate_path
            structure_info['has_train_subdir'] = True
            break
    
    for candidate in test_candidates:
        candidate_path = os.path.join(base_directory, candidate)
        if os.path.isdir(candidate_path):
            test_dir = candidate_path
            structure_info['has_test_subdir'] = True
            break
    
    # Determine structure and suggestions
    if direct_classes:
        structure_info['has_direct_classes'] = True
        structure_info['classes_found'] = direct_classes
        
        if len(direct_classes) >= 2:
            if structure_info['has_train_subdir'] or structure_info['has_test_subdir']:
                structure_info['suggested_mode'] = 'mixed_structure'
                structure_info['warning'] = 'Both direct classes and Train/Test subdirs found. Please clarify structure.'
            else:
                structure_info['suggested_mode'] = 'train_val'
                structure_info['train_dir'] = base_directory
    
    elif structure_info['has_train_subdir']:
        structure_info['train_dir'] = train_dir
        structure_info['test_dir'] = test_dir
        
        if structure_info['has_test_subdir']:
            structure_info['suggested_mode'] = 'train_test'
        else:
            structure_info['suggested_mode'] = 'train_only'
        
        # Check classes in train directory
        train_classes = []
        for class_name in ['original', 'fake', 'real', 'authentic']:
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                train_classes.append(class_name)
        structure_info['classes_found'] = train_classes
    
    return structure_info


# Test the quantum preprocessing
if __name__ == "__main__":
    print("Testing Quantum Image Preprocessing...")
    
    # Test single path
    qprep = QuantumImagePreprocessor(n_qubits=4, n_layers=2)
    dummy_img = tf.random.uniform((2, 128, 128, 1))  # Raw grayscale
    q_features = qprep(dummy_img)
    
    print(f"Input shape: {dummy_img.shape}")
    print(f"Output shape: {q_features.shape}")
    print(f"Trainable weights: {len(qprep.trainable_weights)}")
    
    # Test dual path
    dual_qprep = DualQuantumPreprocessor(n_qubits=4, n_layers=2)
    dual_features = dual_qprep(dummy_img)
    
    print(f"Dual output shape: {dual_features.shape}")
    print(f"Dual trainable weights: {len(dual_qprep.trainable_weights)}")
    print("✓ Quantum preprocessing tests passed!")