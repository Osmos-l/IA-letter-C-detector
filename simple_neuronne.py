w = [0, 0, 0] # Poids des neuronnes
x = [1, 0, 1] # Les entrées

# Cette fonction retourne le signe d'un nombre :
# - 1 si le nombre est positif,
# - -1 si le nombre est négatif ou nul.
def sign(s):
    if s > 0:
        return 1
    else:
        return -1

# Permet de calculer le produit scalaire des valeurs des tableaux w et x
def dot(w, x):
    s = 0

    for i in range(len(w)):
        s += w[i] * x[i]

    return s

def neurone(w, x):
    s = dot(w, x)

    return sign(s)

y = -1 # La réponse attendue
yp = neurone(w, x) # La réponse produite

if (y == yp):
    print("La réponse produite est correcte")
else:
    print("La réponse produit n'est pas correcte")

# Ajustement des poids
e = 0.10 # Taille de la MàJ du poid
for i in range(len(w)):
    print(type(w[i]))  # Ajoute cette ligne pour vérifier le type
    w[i] = w[i] + e * (y - yp) * x[i]

print(w)