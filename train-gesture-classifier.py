# Este programa carga un dataset de gestos de manos (piedra, papel o tijera) capturados previamente, entrena un modelo de 
# deep learning, evalúa su rendimiento y guarda el modelo entrenado para su posterior uso en predicciones en tiempo real.
# -----------------------------------------------------------------------------------------------------------------------

# Librerías utilizadas 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Silenciar mensajes de TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar dataset
print("Cargando dataset...")
try:
    dataset = np.load('rps_dataset.npy', allow_pickle=True).item()
except:
    print("Error al cargar el dataset. Verificar el archivo rps_dataset.npy")
    exit()

# Extraer datos
labels = dataset['etiquetas']
coordinates = dataset['coordenadas']

# Verificar datos
print(f"\nResumen del Dataset:")
print(f"- Muestras totales: {len(labels)}")
print(f"- Forma de coordenadas: {coordinates.shape}")
print(f"- Distribución de clases:")
print(f"  Piedra: {np.sum(labels==0)}")
print(f"  Papel: {np.sum(labels==1)}")
print(f"  Tijeras: {np.sum(labels==2)}\n")

# Preparar datos
coordinates = coordinates.astype('float32')   # Asegura tipo de dato
X = coordinates                               # Asigna las coordenadas de los landmarks a la variable X
y = to_categorical(labels, num_classes=3)     # Transforma etiquetas a one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),   # cada muestra tiene 42 valores
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
print("Iniciando entrenamiento...")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, 
                    validation_split=0.2, verbose=1)

# Guardar modelo
model.save('rps_model.h5')
print("\nModelo guardado en 'rps_model.h5'")

# Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en test: {test_acc:.2f}")

# Gráfico
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Precisión durante entrenamiento')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend()
plt.show()