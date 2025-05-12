import numpy as np
import models.pertubation_gradient as perceptron_perturbation_gradient
import models.partial_derivative_gradient as partial_derivative_gradient
import models.sgd_perceptron as sgd_perceptron
from benchmark import benchmark
from matrices import data
from models_v2.perceptron import Perceptron

# Hyper paramètres
    # All
epochs  = 10000            # Nb d'epochs
weights = np.zeros(25)  # Poids des neuronnes (0 par défaut)
learning_rate = 0.01           # Taille de la MàJ du poid

    # Gradient
dw      = 0.1           # Perturbation

#benchmark(perceptron.learn, epochs, w.copy(), e, data)

#benchmark(perceptron_perturbation_gradient.learn, data, w.copy(), e, dw, epochs)

#benchmark(partial_derivative_gradient.learn, data, w.copy(), e, epochs)

#benchmark(sgd_perceptron.learn, data, w.copy(), e, epochs)

# Create a Perceptron instance
perceptron = Perceptron(epochs, learning_rate, weights.copy())

# Train the perceptron
perceptron.learn(data)