import numpy as np
from TestData import TestData


def sigmoid(x):
    # Activation function: transforms input to value between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of sigmoid function for backpropagation
    return x * (1 - x)

# AND problem
if __name__ == "__main__":
    test_datas = [
        TestData([1, 0], 0),
        TestData([0, 1], 0),
        TestData([1, 1], 1),
        TestData([0, 0], 1)
    ]

    # test_datas = [
    #     TestData([1, 0], 1),
    #     TestData([0, 1], 1),
    #     TestData([1, 1], 0),
    #     TestData([0, 0], 0)
    # ]

    layer1_weights = np.random.rand(2, 2)  # 2 x 2 matrix
    layer1_bias = np.random.rand(2)
    layer2_weights = np.random.rand(2)
    layer2_bias = np.random.rand(1)

    for i in range(1, 5001):
        print(f"--{i:>5}----------------")
        for j, test_data in enumerate(test_datas):

            # output = inputs * weights + bais
            layer1_outputs = np.dot(test_data.input_val, layer1_weights) + layer1_bias
            layer1_outputs_between01 = sigmoid(layer1_outputs)
            layer2_outputs = np.dot(layer1_outputs_between01, layer2_weights) + layer2_bias
            layer2_outputs_between01 = sigmoid(layer2_outputs)[0]

            diff = test_data.output_val - layer2_outputs_between01
            adjustment = diff * sigmoid_derivative(layer2_outputs_between01)
            layer2_weights += 0.2 * adjustment * layer1_outputs_between01
            layer2_bias += 0.2 * adjustment

            diff_hidden = layer2_weights * adjustment
            hidden_adjustment = diff_hidden * sigmoid_derivative(layer1_outputs_between01)
            layer1_weights += 0.2 * hidden_adjustment * test_data.input_val
            layer1_bias += 0.2 * hidden_adjustment

    # predict
    print("\n\n\n\n")
    for test_data in test_datas:
        layer1_outputs = np.dot(test_data.input_val, layer1_weights) + layer1_bias
        layer1_outputs_between01 = sigmoid(layer1_outputs)
        layer2_outputs = np.dot(layer1_outputs_between01, layer2_weights) + layer2_bias
        layer2_outputs_between01 = sigmoid(layer2_outputs)[0]

        print(f"test_data.input_val: {test_data.input_val}, ans: {layer2_outputs_between01}")

    print("\nlayer1_weights: ")
    print(layer1_weights)
    print("layer1_bias: ")
    print(layer1_bias)

    print("\nlayer2_weights: ")
    print(layer2_weights)
    print("layer2_bias: ")
    print(layer2_bias)

    print("")