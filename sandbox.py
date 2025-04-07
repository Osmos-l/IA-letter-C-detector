import numpy as np
import models.perceptron as perceptron
import models.pertubation_gradient as perceptron_perturbation_gradient
import models.partial_derivative_gradient as partial_derivative_gradient
import models.sgd_perceptron as sgd_perceptron
from matrices import data

# Hyper paramètres
    # All
epochs  = 100            # Nb d'epochs
w       = np.zeros(25)  # Poids des neuronnes (0 par défaut)
e       = 0.01           # Taille de la MàJ du poid

    # Gradient
dw      = 0.1           # Perturbation

perceptron.learn(epochs, w.copy(), e, data)
print("\n")
perceptron_perturbation_gradient.learn(data, w.copy(), e, dw, epochs)
print("\n")
partial_derivative_gradient.learn(data, w.copy(), e, epochs)
print("\n")
sgd_perceptron.learn(data, w.copy(), e, epochs)