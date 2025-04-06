import numpy as np
import models.perceptron as perceptron

# Fonction de calcul du gradient par perturbation
def gradient(x, y, w):
    g = np.zeros(len(w)) 

    for i in range(len(w)):
        g[i] = -2 * (y - perceptron.neurone(x, w)) * x[i]

    return g

# Effectuer un pas de gradient
def descend(data, w, e):
    x = np.array(data['matrix']).flatten()
    y = data['is_c']

    g = gradient(x, y, w)
    
    for i in range(len(w)):
        w[i] -= e * g[i]

    return w

# Apprentissage supervisé par gradient sans perturbation
def learn(dataset, w, e, epochs):
    loss_before = perceptron.loss(dataset, w)

    for epoch in range(epochs):
        for idx, data in enumerate(dataset):
            w = descend(data, w, e)
    
    loss_after = perceptron.loss(dataset, w)

    print(f"[Gradient Perceptron] - W: {w}")
    print(f"Loss before: {loss_before}")
    print(f"Loss after: {loss_after}")
