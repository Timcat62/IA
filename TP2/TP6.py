from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).with_name("spambase.csv"))
print(df.head())
print(df.describe())

X = df.drop(columns=["spam"])
y = df["spam"]  # 1 = spam, 0 = courrier normal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

modele = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
modele.fit(X_train, y_train)

print("Score test :", modele.score(X_test, y_test))
predictions = modele.predict(X_test)
print(classification_report(y_test,	predictions))

mat = confusion_matrix(y_test, predictions)
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédiction") ; plt.ylabel("Réalité")
plt.title("Détection des spams")
plt.show()