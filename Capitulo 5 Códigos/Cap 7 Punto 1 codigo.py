# 1. Instalación de dependencias
!pip install pandas numpy scikit-learn tensorflow

# 2. Carga y visualización del dataset
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())
print(df['species'].value_counts())  # 50 de cada clase

# 3. Separar X e y, y codificar
X = iris.data
y = iris.target  # 0,1,2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# 4. Definir red neuronal
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Entrenar
history = model.fit(X_train, y_train_cat,
                    validation_split=0.2,
                    epochs=100, batch_size=8, verbose=1)

# 6. Evaluar en test
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nAccuracy en test: {acc:.4f}")

# 7. Probar algunas predicciones
import numpy as np
for i in range(5):
    xi = X_test[i:i+1]
    yi = np.argmax(y_test_cat[i])
    pred = np.argmax(model.predict(xi))
    print(f"Muestras {i}: real={iris.target_names[yi]}, predicción={iris.target_names[pred]}")
