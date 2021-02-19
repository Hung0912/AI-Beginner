import numpy as np


class NeuralNetwork():
    def __init__(self):
        # random weight tu -1 den 1
        self.weight = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, input):
        input = input.astype(float)
        output = self.sigmoid(np.dot(input, self.weight))
        return output

    def train(self, training_input, training_output, training_iteration):

        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iteration):
            # siphon the training data via  the neuron
            output = self.think(training_input)

            # computing error rate for back-propagation
            error = training_output - output

            # performing weight adjustments
            adjustment = np.dot(training_input.T, error *
                                self.sigmoid_derivative(output))

            self.weight += adjustment


if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.weight)

    # training data consisting of 4 examples--3 input values and 1 output
    training_input = np.array([[0, 0, 1],
                               [1, 1, 1],
                               [1, 0, 1],
                               [0, 1, 1]])

    training_output = np.array([[0, 1, 1, 0]]).T

    # training taking place
    neural_network.train(training_input, training_output, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.weight)

    user_input_one = str(input("User Input One: "))
    user_input_two = str(input("User Input Two: "))
    user_input_three = str(input("User Input Three: "))

    print("Considering New Situation: " + user_input_one +
          user_input_two + user_input_three)
    print("New Output data: ")
    print(neural_network.think(
        np.array([user_input_one, user_input_two, user_input_three])))
    print("Wow, we did it!")
