import time
import numpy as np

class XorNeuralNetwork:
    def __init__(self, learning_rate=0.2):
        # Neural network with one hidden layer containing 2 neurons
        # Initialize with random weights
        self.hidden_weights = np.random.rand(2, 2)  # 2 inputs -> 2 hidden neurons
        self.hidden_bias = np.random.rand(2)  # bias for hidden layer
        self.output_weights = np.random.rand(2)  # 2 hidden neurons -> 1 output
        self.output_bias = np.random.rand(1)  # bias for output
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        # Activation function: transforms input to value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of sigmoid function for backpropagation
        return x * (1 - x)

    def predict(self, inputs):
        # Forward pass
        # Hidden layer
        sums = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        hidden_outputs = self.sigmoid(sums)

        # hidden_outputs as input to output layer
        sum_final = np.dot(hidden_outputs, self.output_weights) + self.output_bias
        predicted_value = self.sigmoid(sum_final)

        return predicted_value[0], hidden_outputs

    def train(self, training_inputs, training_outputs, epochs=5000):
        for epoch in range(epochs):
            sum_error = 0

            # For each training example
            for inputs, expected_output in zip(training_inputs, training_outputs):
                # Step 1: Predict (forward pass)
                prediction, hidden_outputs = self.predict(inputs)

                # Step 2: Calculate error
                error = expected_output - prediction
                sum_error += np.abs(error)

                # Step 3: Adjust weights and biases (backpropagation)
                adjustment = error * self.sigmoid_derivative(prediction)
                self.output_weights += self.learning_rate * adjustment * hidden_outputs
                self.output_bias += self.learning_rate * adjustment

                # distributes error back to each hidden neuron based on its weight
                hidden_error = self.output_weights * adjustment
                hidden_adjustment = hidden_error * self.sigmoid_derivative(hidden_outputs)
                self.hidden_weights += self.learning_rate * hidden_adjustment * inputs
                self.hidden_bias += self.learning_rate * hidden_adjustment

            # Print progress every 1000 epochs
            if (epoch + 1) % 100 == 0:
                print(f"\rEpoch {epoch+1:>5}, Error: {sum_error:.4f}", end='', flush=True)
            time.sleep(0.00025)

        print("\nTraining complete.")