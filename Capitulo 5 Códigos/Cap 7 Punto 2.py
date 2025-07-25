pip install pandas numpy scikit-learn tensorflow

# 1. Importar librerías necesarias
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout
)
from tensorflow.keras.utils import to_categorical

# 2. Descargar el dataset Fashion MNIST desde TensorFlow
#    (60 000 imágenes de entrenamiento, 10 000 de prueba; cada imagen 28×28 en escala de grises)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 3. Preprocesamiento
#    a) Normalizar los píxeles a rango [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

#    b) Añadir canal de color para Conv2D: (28,28) → (28,28,1)
x_train = x_train[..., tf.newaxis]
x_test  = x_test[..., tf.newaxis]

#    c) Codificar las etiquetas en one‑hot (10 clases)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat  = to_categorical(y_test,  num_classes=10)

# 4. Definir la arquitectura de la red neuronal convolucional
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 clases de ropa
])

# 5. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Entrenar la red
history = model.fit(
    x_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 7. Evaluar en el conjunto de prueba
loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f'\nPrecisión en test: {acc*100:.2f}%')

# 8. Probar algunas predicciones
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
import numpy as np

for i in range(5):
    img = x_test[i:i+1]                  # tomar la i‑ésima muestra
    pred = model.predict(img)           # obtener probabilidades
    label = np.argmax(pred, axis=1)[0]  # clase predicha
    real  = y_test[i]                   # etiqueta real
    print(f'Muestra {i}: real = {class_names[real]}, pred = {class_names[label]}')


# 1. Mostrar la historia de entrenamiento
print("Última pérdida de entrenamiento:", history.history['loss'][-1])
print("Última precisión de entrenamiento:", history.history['accuracy'][-1])
print("Última pérdida de validación:", history.history['val_loss'][-1])
print("Última precisión de validación:", history.history['val_accuracy'][-1])

# 2. Evaluar en test y mostrar métricas detalladas
loss_test, acc_test = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nPérdida en test: {loss_test:.4f}")
print(f"Precisión en test: {acc_test*100:.2f}%")

# 3. Reporte de clasificación (usando sklearn)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 3.1. Obtener predicciones de clase
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# 3.2. Mostrar informe y matriz de confusión
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# 4. Ejemplos de predicción con su probabilidad
for i in range(5):
    probs = y_pred_probs[i]
    pred = y_pred[i]
    real = y_true[i]
    print(f"\nMuestra {i}:")
    print(f"  Etiqueta real: {class_names[real]}")
    print(f"  Predicción:    {class_names[pred]}")
    print(f"  Probabilidades: {np.round(probs, 3)}")
