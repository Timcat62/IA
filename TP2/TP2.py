import numpy as np

def perceptron(entrees, poids, biais):
    z = np.dot(entrees, poids) + biais
    return 1 if z >= 0 else 0

poids = np.array([1, 1])
biais = -0.5

print(perceptron(np.array([0, 0]), poids, biais))
print(perceptron(np.array([1, 0]), poids, biais))
print(perceptron(np.array([0, 1]), poids, biais))
print(perceptron(np.array([1, 1]), poids, biais))