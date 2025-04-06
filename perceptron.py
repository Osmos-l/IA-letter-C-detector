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

def simple_learning(epochs, w, e, data): 
    for epoch in range(epochs):
        print(f"Epoch n°{epoch +1}")

        for idx, entry in enumerate(data):
            x = np.array(entry['matrix']).flatten()

            y = entry['is_c'] # La réponse désirée
            yp = neurone(w, x) # La réponse produite
        
            # Ajustement des poids
            for i in range(len(w)):
                w[i] = w[i] + e * (y - yp) * x[i]

    return w