import time
import numpy as np


def activate(x):
    # Activation function: transforms input to value between 0 and 1
    # using Sigmoid function
    return 1 / (1 + np.exp(-x))


def activate_fun_derivative(x):
    # Derivative of sigmoid function for backpropagation
    # Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    return x * (1 - x)


class SimpleNeuralNetwork:
    def __init__(self, learning_rate=0.2):
        # Just one neuron with 2 inputs
        # Initialize with random weights
        self.weights = np.random.rand(2) # how important each input is. 0 ~ 1, e.g., [0.64459, 0.38052]
        self.bias = np.random.rand(1)    # how much to shift the output. 0 ~ 1, e.g., 0.957732
        self.learning_rate = learning_rate

    def train(self, training_inputs, training_outputs, epochs=2000):
        for epoch in range(epochs):
            # For each training example
            # zip([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
            #   → [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
            for inputs, expected_output in zip(training_inputs, training_outputs):
                # Step 1: Predict (forward pass)
                prediction = self.predict(inputs)

                # Step 2: Calculate error
                error = expected_output - prediction

                # Step 3: Adjust weights and bias (backpropagation)
                adjustment = error * activate_fun_derivative(prediction)
                self.weights += self.learning_rate * adjustment * inputs
                self.bias += self.learning_rate * adjustment

            # Print progress every 100 epochs
            # if epoch % 100 == 0:
            #     total_error = 0
            #     for inputs, expected_output in zip(training_inputs, training_outputs):
            #         total_error += abs(expected_output - self.predict(inputs))
            #     print(f"Epoch {epoch}, Error: {total_error}")

            # Print bias, weights every 10 epochs
            print(f"\rEpoch {epoch+1:>3}, "
                  f"Bias: {self.bias[0]:>5.2f}, "
                  f"Adjust: {adjustment:>5.4f}, "
                  f"Error: {abs(expected_output - prediction):>5.3f}, "
                  f"SigDeri: {activate_fun_derivative(prediction):>5.3f}, "
                  f"Weights: {self.weights[0]:>5.2f}, {self.weights[1]:>5.2f}", end='', flush=True)

            time.sleep(0.001)
        print("\nTraining complete.")

    def predict(self, inputs):
        # Forward pass: calculate the weighted sum, then apply activation
        weight_sum = np.dot(inputs, self.weights) + self.bias
        predicted_value = activate(weight_sum)
        return predicted_value[0]