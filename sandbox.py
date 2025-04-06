import numpy as np
import perceptron
from matrices import data

# Hyper paramètres
    # All
epochs  = 10            # Nb d'epochs
w       = np.zeros(25)  # Poids des neuronnes (0 par défaut)
e       = 0.1           # Taille de la MàJ du poid

    # Gradient
dw      = 0.1           # Perturbation

loss_before_simple_learning = perceptron.loss(data, w)

#w   = perceptron.simple_learn(epochs, w, e, data)
w  = perceptron.gradient_perturbation_learn(data, w, e, dw, epochs)

loss_after_simple_learning = perceptron.loss(data, w)

print(f"W: {w}")
print(f"Loss before: {loss_before_simple_learning}")
print(f"Loss after: {loss_after_simple_learning}")