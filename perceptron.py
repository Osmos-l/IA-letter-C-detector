import numpy as np

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

# Mesure la distance entre la sortie produite et la sortie désirée pour un exemple
def cost(x, y, w):
    yp = neurone(w, x)
    c  = (y - yp) ** 2

    return c

# Fonction de cacul du coût moyen sur un ensemble d'exemples
def loss(data, w):
    P = len(data)
    s = 0

    for idx, entry in enumerate(data):
        x = np.array(entry['matrix']).flatten()
        y = entry['is_c']

        s += cost(x, y, w)

    return s / P

# Fonction de calcul du gradient par perturbation
def gradient(data, w, dw):
    h = loss(data, w)
    g = np.zeros(len(w)) 

    for i in range(len(w)):
        wa = w
        wa[i] = w[i] + dw
        a = loss(data, wa)
        g[i] = (a - h) / dw

    return g

# Effectuer un pas de gradient
def descend(data, w, e, dw):
    g = gradient(data, w, dw)
    
    for i in range(len(w)):
        w[i] -= e * g[i]

    return w

def gradient_perturbation_learn(data, w, e, dw, epochs):
    for i in range(epochs):
        w = descend(data, w, e, dw)
    
    return w

def simple_learn(epochs, w, e, data): 
    for epoch in range(epochs):
        #print(f"Epoch n°{epoch +1}")

        for idx, entry in enumerate(data):
            x = np.array(entry['matrix']).flatten()

            y = entry['is_c'] # La réponse désirée
            yp = neurone(w, x) # La réponse produite
        
            # Ajustement des poids
            for i in range(len(w)):
                w[i] = w[i] + e * (y - yp) * x[i]

    return w