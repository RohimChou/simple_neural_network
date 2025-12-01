import numpy as np
from DosageData import DosageData


def sigmoid(x):
    # Activation function: transforms input to value between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of sigmoid function for backpropagation
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Dosage effectiveness prediction
if __name__ == "__main__":
    # Sample training data: (dosage in mg, effectiveness)
    # inputs: dosage (scalar, between 0 and 10 mg)
    # outputs: effectiveness (between 0 and 1)
    dosage_datas = [
        DosageData(0, 0),
        DosageData(1, 0.2),
        DosageData(3, 0.9),
        DosageData(5, 1),
        DosageData(7, 0.7),
        DosageData(10, 0.1),
    ]

    np.random.seed(42)
    learning_rate = 0.2
    epochs = 10000
    input_size = 1
    hidden_size = 2

    layer1_weights = np.random.rand(input_size, hidden_size) # 1 x 2 matrix
    layer1_bias = np.random.rand(1, hidden_size)
    layer2_weights = np.random.rand(hidden_size, 1) # 2 x 1 matrix
    layer2_bias = np.random.rand(1, 1)

    for epoch in range(epochs):
        for data in dosage_datas:
            # Forward pass
            data_inputs = np.array([[data.dosage / 10]]) # normalize input
            layer1_outputs = np.dot(data_inputs, layer1_weights) + layer1_bias
            layer1_outputs_between01 = sigmoid(layer1_outputs)

            layer2_outputs = np.dot(layer1_outputs_between01, layer2_weights) + layer2_bias
            layer2_outputs_between01 = sigmoid(layer2_outputs)

            # Backward pass
            # axis=0: Sum down the rows (i.e., sum each column)
            # keepdims=True: Do not remove the axis being summed — keep the result as a 2D array instead of collapsing it
            diff = data.effectiveness - layer2_outputs_between01
            # The derivative determines how sensitive the output is to
            # changes in its input. If the output is near 0 or 1 (saturated),
            # the derivative is near zero, and the adjustment will be small,
            # limiting the weight update.
            adjustment = diff * sigmoid_derivative(layer2_outputs_between01)
            layer2_weights += layer1_outputs_between01.T.dot(adjustment) * learning_rate
            layer2_bias += np.sum(adjustment, axis=0, keepdims=True) * learning_rate

            diff_layer1 = adjustment.dot(layer2_weights.T)
            adjustment_layer1 = diff_layer1 * sigmoid_derivative(layer1_outputs_between01)
            layer1_weights += data_inputs.T.dot(adjustment_layer1) * learning_rate
            layer1_bias += np.sum(adjustment_layer1, axis=0, keepdims=True) * learning_rate

    # Test the trained model
    print("\nlayer1_weights: ")
    print(layer1_weights)
    print("layer1_bias: ")
    print(layer1_bias)

    print("\nlayer2_weights: ")
    print(layer2_weights)
    print("layer2_bias: ")
    print(layer2_bias)

    print("")
    for data in dosage_datas:
        data_inputs = np.array([[data.dosage / 10]])
        layer1_outputs = np.dot(data_inputs, layer1_weights) + layer1_bias
        layer1_outputs_between01 = sigmoid(layer1_outputs)

        layer2_outputs = np.dot(layer1_outputs_between01, layer2_weights) + layer2_bias
        layer2_outputs_between01 = sigmoid(layer2_outputs)
        print(f"Input Dosage: {data_inputs[0][0]} mg, Predicted Effectiveness: {layer2_outputs_between01[0][0]:.4f}, Actual Effectiveness: {data.effectiveness}")