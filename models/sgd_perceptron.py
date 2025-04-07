import numpy as np
import random
import models.perceptron as perceptron

# Fonction de calcul du gradient, dérivée partielle
def gradient(x, y, w):
    g = np.zeros(len(w)) 

    for i in range(len(w)):
        g[i] = -2 * (y - perceptron.neurone(x, w)) * x[i]

    return g

# Effectuer un pas de gradient
def descend(w, e, g):
    for i in range(len(w)):
        w[i] -= e * g[i]

    return w

# Apprentissage supervisé par gradient stochastique
def learn(dataset, w, e, epochs):
    loss_before = perceptron.loss(dataset, w)

    p = len(dataset)
    for epoch in range(epochs):
        k = random.randrange(0, p)
        
        x = np.array(dataset[k]['matrix']).flatten()
        y = dataset[k]['is_c']

        g = gradient(x, y, w)

        w = descend(w, e, g)
    
    loss_after = perceptron.loss(dataset, w)

    print(f"[SGD Perceptron] - W: {w}")
    print(f"Loss before: {loss_before}")
    print(f"Loss after: {loss_after}")
