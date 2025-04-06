import numpy as np
import models.perceptron as perceptron
import models.pertubation_gradient as perceptron_perturbation_gradient
import models.gradient as perceptron_gradient
from matrices import data

# Hyper paramètres
    # All
epochs  = 10            # Nb d'epochs
w       = np.zeros(25)  # Poids des neuronnes (0 par défaut)
e       = 0.1           # Taille de la MàJ du poid

    # Gradient
dw      = 0.1           # Perturbation

perceptron.learn(epochs, w.copy(), e, data)
print("\n")
perceptron_perturbation_gradient.learn(data, w.copy(), e, dw, epochs)
print("\n")
perceptron_gradient.learn(data, w.copy(), e, epochs)