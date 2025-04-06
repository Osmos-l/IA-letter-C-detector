import numpy as np
import models.perceptron as perceptron

# Fonction de calcul du gradient par perturbation
def gradient(data, w, dw):
    h = perceptron.loss(data, w)
    g = np.zeros(len(w)) 

    for i in range(len(w)):
        wa = w
        wa[i] = w[i] + dw
        a = perceptron.loss(data, wa)
        g[i] = (a - h) / dw

    return g

# Effectuer un pas de gradient
def descend(data, w, e, dw):
    g = gradient(data, w, dw)
    
    for i in range(len(w)):
        w[i] -= e * g[i]

    return w

# Apprentissage supervisé par gradient avec perturbation
def learn(data, w, e, dw, epochs):
    loss_before = perceptron.loss(data, w)

    for i in range(epochs):
        w = descend(data, w, e, dw)
    
    loss_after = perceptron.loss(data, w)

    print(f"[Perturbation Gradient Perceptron] - W: {w}")
    print(f"Loss before: {loss_before}")
    print(f"Loss after: {loss_after}")
