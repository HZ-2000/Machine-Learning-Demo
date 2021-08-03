#------------------------------------------#
# Creator: Zach Heimbigner                 #
# Last Modified: 7/23/2021                 #
# Language: Python 3                       #
# Command: NA                              #
# Notes:                                   #
#  - zip issue corrected                   #
#  - Neuron class variable weights changed #
#    to an array for scalability           #
#------------------------------------------#

from numpy import exp, array, random, dot, apply_along_axis

class Neuron:
    def __init__(self, num_inputs):
        self.weights = random.normal(size=(num_inputs,))
        self.bias = random.normal()

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def deriv_sigmoid(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

class NeuralNetwork:
    def __init__(self, num_neurons):
        self.output = Neuron(num_neurons)
        self.hidden = []
        for i in range(num_neurons):
            self.hidden.append(Neuron(num_neurons))

    def feed_forward(self, x):
        results = []
        for neuron in self.hidden:
            results.append(neuron.sigmoid(dot(x, neuron.weights) + neuron.bias))

        return self.output.sigmoid(dot(results, self.output.weights) + self.output.bias)

    def train(self, data, targets):
        # this function trains the network
        # it uses 2 arrays to keep track of
        # sums and values for training so that
        # any number of neurons can be added
        learn_rate = 0.1
        iterations = 500

        for iteration in range(iterations):
            for x, y_true in zip(data, targets):

                #feed forward
                sums = []
                vals = []
                for neuron in self.hidden:
                    sum = dot(neuron.weights, x) + neuron.bias
                    sums.append(sum)
                    vals.append(neuron.sigmoid(sum))

                sum_o1 = dot(self.output.weights, vals) + self.output.bias
                o1 = self.output.sigmoid(sum_o1)
                y_pred = o1

                #partial l / partial w1
                dld_ypred = -2 * (y_true - y_pred)

                #output layer
                d_ypred_d_h = []
                for weight in self.output.weights:
                    d_ypred_d_h.append(weight * self.output.deriv_sigmoid(sum_o1))

                for val, i in zip(vals, range(len(vals))):
                    d_ypred_d_ow = val * self.output.deriv_sigmoid(sum_o1)
                    self.output.weights[i] -= learn_rate * dld_ypred * d_ypred_d_ow

                d_ypred_d_ob = self.output.deriv_sigmoid(sum_o1)
                self.output.bias -= learn_rate * dld_ypred * d_ypred_d_ob

                # hidden layer
                for neuron, i in zip(self.hidden, range(len(self.hidden))):
                    for val, k in zip(x, range(len(x))):
                        d_h_d_w = val * neuron.deriv_sigmoid(sums[i])
                        neuron.weights[k] -= learn_rate * dld_ypred * d_ypred_d_h[i] * d_h_d_w

                    neuron.bias -= learn_rate * dld_ypred * d_ypred_d_h[i] * neuron.deriv_sigmoid(sums[i])
