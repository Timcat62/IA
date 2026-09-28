from tensorflow import keras
from tensorflow.keras import layers

modele = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # aplatit l'image en 784 entrées
    layers.Dense(128, activation="relu"),    # couche cachée de 128 neurones
    layers.Dense(64, activation="relu"),     # couche cachée de 64 neurones
    layers.Dense(10, activation="softmax")   # 10 sorties, une par catégorie
])