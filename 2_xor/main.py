import numpy as np
from xor_neural_network import XorNeuralNetwork

# XOR problem
if __name__ == "__main__":
    # XOR logic gate: inputs must be different for output to be 1
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 1, 1, 0])

    # Create and train the neural network
    nn = XorNeuralNetwork()
    nn.train(training_inputs, training_outputs)

    # Test the trained network
    print("\nTesting the trained neural network on XOR logic:")
    for inputs in training_inputs:
        prediction, _ = nn.predict(inputs)
        print(f"Inputs: {inputs}, Prediction: {prediction:.4f}, Rounded: {round(prediction)}")