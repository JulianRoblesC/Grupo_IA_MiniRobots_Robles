import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. Cargar y explorar el dataset Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Mostrar las primeras filas
print("Vista previa del dataset Wine:")
print(df.head())

# 2. Características y rótulo
print("\nCaracterísticas (features):", wine.feature_names)
print("Rótulos (target names):", wine.target_names)

# 3. Preprocesamiento
X = wine.data
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Definir y entrenar la red neuronal (MLPClassifier de scikit-learn)
model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluación
acc = model.score(X_test, y_test)
print(f"\nPrecisión en test: {acc*100:.2f}%")

# 6. Ejemplos basados en pesos aprendidos para varias muestras
weights = model.coefs_[0]
biases = model.intercepts_[0]

print("\nEjemplos de salidas de la primera capa y predicciones:")
for sample_idx in range(5):
    x_sample = X_test[sample_idx]
    # Salida manual de la primera capa con ReLU
    first_layer_output = np.maximum(0, np.dot(x_sample, weights) + biases)
    # Predicción de probabilidades y clase
    probs = model.predict_proba(x_sample.reshape(1, -1))[0]
    pred_class = wine.target_names[np.argmax(probs)]
    real_class = wine.target_names[y_test[sample_idx]]

    print(f"\nMuestra {sample_idx}:")
    print(f"  Salida capa 1 (5 primeros valores): {np.round(first_layer_output[:5], 3)}")
    print(f"  Probabilidades: {np.round(probs, 3)}")
    print(f"  Clase real: {real_class}, Clase predicha: {pred_class}")
