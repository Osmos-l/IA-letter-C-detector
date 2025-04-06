import numpy as np
from perceptron import simple_learning
from matrices import data

# Hyper paramètres
epochs  = 10         # Nb d'epochs
w       = np.zeros(25)  # Poids des neuronnes (0 par défaut)
e       = 0.1           # Taille de la MàJ du poid

w = simple_learning(epochs, w, e, data)
print(w)