# Baseline CNN for Quantum Circuit Classification

This project implements a baseline Convolutional Neural Network (CNN) for classifying data related to quantum circuits. The repository includes scripts for training, evaluating, and saving the CNN model using Keras.

## Project Structure

- `cnn_model.py` — Defines the CNN architecture and model creation functions.
- `train_cnn.py` — Script to train the CNN model on your dataset.
- `evaluate_cnn.py` — Script to evaluate the trained model's performance.
- `cnn_model.keras` — Saved Keras model file (generated after training).

## Getting Started

### Prerequisites
- Python 3.7+
- Keras
- TensorFlow
- NumPy

Install dependencies with:
```bash
pip install tensorflow keras numpy
```

### Training the Model
Run the following command to train the CNN model:
```bash
python train_cnn.py
```

### Evaluating the Model
After training, evaluate the model using:
```bash
python evaluate_cnn.py
```

## Model File
- The trained model is saved as `cnn_model.keras` after running the training script.

## Notes
- Ensure your dataset is properly formatted as expected by the scripts.
- Modify the scripts as needed for your specific data or experiment.

## License
This project is provided for educational and research purposes.
