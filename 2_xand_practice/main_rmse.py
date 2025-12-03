import numpy as np
from TestData import TestData


def sigmoid(x):
    # Activation function: transforms input to value between 0 and 1
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # Derivative of sigmoid function for backpropagation
    return x * (1 - x)


def calculate_rmse(predictions, targets):
    # Calculate Root Mean Square Error
    return np.sqrt(np.mean((predictions - targets) ** 2))


# AND problem ⭐⭐ DISPLAY RMSE AS LOSS FUNCTION ⭐⭐
if __name__ == "__main__":
    test_datas = [
        TestData([1, 0], 0),
        TestData([0, 1], 0),
        TestData([1, 1], 1),
        TestData([0, 0], 1)
    ]

    layer1_weights = np.random.rand(2, 2)  # 2 x 2 matrix
    layer1_bias = np.random.rand(2)
    layer2_weights = np.random.rand(2)
    layer2_bias = np.random.rand(1)

    learning_rate = 0.2

    for epoch in range(1, 5001):
        epoch_predictions = []
        epoch_targets = []

        for test_data in test_datas:
            # Forward pass
            layer1_outputs = np.dot(test_data.input_val, layer1_weights) + layer1_bias
            layer1_outputs_between01 = sigmoid(layer1_outputs)
            layer2_outputs = np.dot(layer1_outputs_between01, layer2_weights) + layer2_bias
            layer2_outputs_between01 = sigmoid(layer2_outputs)[0]

            # Store predictions and targets for RMSE calculation
            epoch_predictions.append(layer2_outputs_between01)
            epoch_targets.append(test_data.output_val)

            # Backward pass
            error = test_data.output_val - layer2_outputs_between01

            # Layer 2 gradients
            layer2_delta = error * sigmoid_derivative(layer2_outputs_between01)
            layer2_weights += learning_rate * layer2_delta * layer1_outputs_between01
            layer2_bias += learning_rate * layer2_delta

            # Layer 1 gradients
            layer1_error = layer2_weights * layer2_delta
            layer1_delta = layer1_error * sigmoid_derivative(layer1_outputs_between01)
            layer1_weights += learning_rate * layer1_delta * np.array(test_data.input_val).reshape(-1, 1)
            layer1_bias += learning_rate * layer1_delta

        # Calculate and print RMSE every 500 epochs
        if epoch % 500 == 0:
            rmse = calculate_rmse(np.array(epoch_predictions), np.array(epoch_targets))
            print(f"Epoch {epoch:>5} - RMSE: {rmse:.6f}")

    # Predict and evaluate final RMSE
    print("\n" + "=" * 50)
    print("Final Predictions:")
    print("=" * 50)

    final_predictions = []
    final_targets = []

    for test_data in test_datas:
        layer1_outputs = np.dot(test_data.input_val, layer1_weights) + layer1_bias
        layer1_outputs_between01 = sigmoid(layer1_outputs)
        layer2_outputs = np.dot(layer1_outputs_between01, layer2_weights) + layer2_bias
        layer2_outputs_between01 = sigmoid(layer2_outputs)[0]

        final_predictions.append(layer2_outputs_between01)
        final_targets.append(test_data.output_val)

        print(
            f"Input: {test_data.input_val} | Target: {test_data.output_val} | Prediction: {layer2_outputs_between01:.6f}")

    final_rmse = calculate_rmse(np.array(final_predictions), np.array(final_targets))
    print(f"\nFinal RMSE: {final_rmse:.6f}")

    print("\n" + "=" * 50)
    print("Learned Parameters:")
    print("=" * 50)
    print("\nLayer 1 Weights:")
    print(layer1_weights)
    print("\nLayer 1 Bias:")
    print(layer1_bias)
    print("\nLayer 2 Weights:")
    print(layer2_weights)
    print("\nLayer 2 Bias:")
    print(layer2_bias)