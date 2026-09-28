import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- ÉTAPE 1 : charger ---
df = pd.read_csv("spambase.csv")     # adapte le nom
print(df.head())
print(df.describe())

# --- ÉTAPE 2 : prétraiter (à adapter à TES colonnes) ---
# df = df.dropna()
# df["colonne_texte"] = df["colonne_texte"].map({"valeurA": 0, "valeurB": 1})

# --- ÉTAPE 3 : séparer X / y puis train / test ---
X = df.drop(columns=["cible"])          # remplace "cible" par ta colonne à prédire
y = df["cible"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ÉTAPE 4 : normaliser + entraîner ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

modele = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
modele.fit(X_train, y_train)

# --- ÉTAPE 5 : évaluer ---
print("Score test :", modele.score(X_test, y_test))
predictions = modele.predict(X_test)
print(classification_report(y_test, predictions))

mat = confusion_matrix(y_test, predictions)
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédiction"); plt.ylabel("Réalité")
plt.title("Matrice de confusion - mon projet")
plt.show()