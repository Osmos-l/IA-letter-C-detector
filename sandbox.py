import numpy as np
from perceptron import simple_learning, loss
from matrices import data

# Hyper paramètres
epochs  = 10000         # Nb d'epochs
w       = np.zeros(25)  # Poids des neuronnes (0 par défaut)
e       = 0.1           # Taille de la MàJ du poid

loss_before_simple_learning = loss(data, w)

w = simple_learning(epochs, w, e, data)

loss_after_simple_learning = loss(data, w)

print(f"W: {w}")
print(f"Loss before: {loss_before_simple_learning}")
print(f"Loss after: {loss_after_simple_learning}")