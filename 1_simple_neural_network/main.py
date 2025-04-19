import numpy as np
from simple_neural_network import SimpleNeuralNetwork

if __name__ == "__main__":
    # AND logic gate: both inputs must be 1 for output to be 1
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 0, 0, 1])

    # Create and train the neural network
    nn = SimpleNeuralNetwork()
    nn.train(training_inputs, training_outputs)

    # Test the trained network
    print("\nTesting the trained neural network on AND logic:")
    for inputs in training_inputs:
        prediction = nn.predict(inputs)
        print(f"Inputs: {inputs}, Prediction: {prediction:.4f}, Rounded: {round(prediction)}")