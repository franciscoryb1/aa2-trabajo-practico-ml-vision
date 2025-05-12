# 🎲 Sistema de Reconocimiento de Gestos - Piedra, Papel o Tijera

Este conjunto de scripts permite grabar gestos manuales, entrenar un modelo de clasificación y utilizarlo en tiempo real para reconocer los gestos de piedra, papel o tijera mediante la cámara web.

---

## 📌 `record-dataset.py`  
### ⚙️ ¿Para qué sirve?  
Este script permite grabar un dataset de gestos utilizando la cámara web.

### 🔧 Modo de uso (solo para el desarrollador):  
1. Ejecutar el script para abrir la cámara.  
2. Realizar un gesto (piedra, papel o tijera).  
3. Presionar la tecla correspondiente:  
   - `0` → Guarda como "piedra"  
   - `1` → Guarda como "papel"  
   - `2` → Guarda como "tijera"  
4. Presionar `q` para salir y guardar el dataset como `rps_dataset.npy`.

### ℹ️ Importante  
- Este paso ya fue realizado durante el desarrollo.  
- El dataset final, incluido en este repositorio, contiene 50 muestras por clase.  
- **El usuario no necesita ejecutar este script.**

---

## 📌 `train-gesture-classifier.py`  
### ⚙️ ¿Para qué sirve?  
Este script entrena un modelo de red neuronal para reconocer los gestos de piedra, papel o tijera a partir del dataset grabado previamente.

### 🔧 Modo de uso (solo para el desarrollador):  
Al ejecutar el script, se realiza automáticamente lo siguiente:  
- Carga el archivo `rps_dataset.npy`.  
- Entrena un modelo de clasificación gestual.  
- Muestra la precisión alcanzada.  
- Guarda el modelo entrenado como `rps_model.h5`.  
- Genera un gráfico con la evolución de la precisión.

### ℹ️ Importante  
- Este modelo ya fue entrenado previamente.  
- El archivo `rps_model.h5` está incluido en el repositorio.  
- **El usuario no necesita ejecutar este script.**

---

## 📌 `rock-paper-scissors.py`  
### ⚙️ ¿Para qué sirve?  
Este es el script principal que permite al usuario interactuar con el sistema.

### 🧑‍💻 Modo de uso (para el usuario):  
1. Ejecutar el script para activar el reconocimiento gestual en tiempo real.  
2. El sistema automáticamente:  
   - Detecta la mano usando **MediaPipe**  
   - Clasifica el gesto (piedra, papel o tijera) utilizando el modelo entrenado  
   - Muestra el resultado en pantalla  

### 🎮 Controles  
- `q` → Salir del programa

### ⚠️ Requisitos previos  
- El archivo `rps_model.h5` debe estar en el mismo directorio que el script  
- La cámara web debe estar disponible y funcional

### ℹ️ Importante  
**Este es el único script que debe ejecutar el usuario.**
