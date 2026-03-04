import tensorflow as tf
import pennylane as qml
import numpy as np

n_qubits = 4
n_layers = 1

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qc(inputs, weights):
    # inputs expected shape (..., n_qubits)
    # constant pi inside circuit ensures consistent dtype
    pi = tf.constant(np.pi, dtype=inputs.dtype)
    for i in range(n_qubits):
        qml.RX(inputs[..., i] * pi, wires=i)
        qml.RY(inputs[..., i] * pi, wires=i)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

w_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
w = tf.random.normal(w_shape)
x = tf.random.uniform((8, 4))
print("wshape", w_shape)
print("output", qc(x, w))
