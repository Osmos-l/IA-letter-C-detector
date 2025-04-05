import numpy as np
from matrices import data

def sign(s):
    if s > 0:
        return 1
    else:
        return -1

def dot(w, x):
    s = 0

    for i in range(len(w)):
        s += w[i] * x[i]

    return s

def neurone(w, x):
    s = dot(w, x)

    return sign(s)

# HyperParamétres
epochs  = 10000         # Nb d'epochs
w       = np.zeros(25)  # Poids des neuronnes
e       = 0.1           # Taille de la MàJ du poid

for epoch in range(epochs):
    print(f"Epoch n°{epoch}")

    for idx, entry in enumerate(data):
        x = np.array(entry['matrix']).flatten()

        y = entry['is_c'] # La réponse désirée
        yp = neurone(w, x) # La réponse produite
    
        # Ajustement des poids
        for i in range(len(w)):
            w[i] = w[i] + e * (y - yp) * x[i]

print(w)