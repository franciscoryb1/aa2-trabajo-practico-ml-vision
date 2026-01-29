# Trabajo Práctico de Aprendizaje Automático II — Modelos Supervisados y Visión por Computadora

Repositorio que contiene el desarrollo del **Trabajo Práctico 1 de Aprendizaje Automático II**, organizado en tres problemas independientes.

Se abordan tareas de regresión y clasificación utilizando modelos clásicos y redes neuronales, así como un flujo completo de **visión por computadora con detección de gestos en tiempo real**.

El enfoque es académico, con una presentación profesional que documenta objetivos, datos y tecnologías utilizadas en cada módulo.

---

## Tecnologías utilizadas

* **Python 3**
* **NumPy** y **Matplotlib** (preprocesamiento y visualización)
* **TensorFlow / Keras** (redes neuronales)
* **scikit-learn** (modelos clásicos y métricas)
* **OpenCV** y **MediaPipe** (visión por computadora y reconocimiento gestual)
* **Jupyter Notebook** (análisis reproducible)

> Nota: las dependencias exactas pueden variar por problema; se recomienda crear un entorno virtual e instalar los paquetes necesarios antes de ejecutar los notebooks o scripts.

---

## Estructura del repositorio

* **Problema_1/**: predicción de desempeño académico a partir de variables de estudio y contexto.
* **Problema_2/**: sistema de reconocimiento gestual para el juego “piedra, papel o tijera”.
* **Problema_3/**: clasificación de escenas naturales con una red neuronal convolucional (CNN).

---

## Problema 1 — Predicción de desempeño académico

**Objetivo:** modelar el *Performance Index* de estudiantes en función de variables como horas de estudio, puntajes previos y hábitos.

### Datos

* Archivo: `Student_Performance.csv`
* Columnas:

  * `Hours Studied`
  * `Previous Scores`
  * `Extracurricular Activities`
  * `Sleep Hours`
  * `Sample Question Papers Practiced`
  * `Performance Index`

### Modelos principales

* Regresión lineal (baseline).
* Red neuronal *feed-forward* (mejor desempeño en validación y test).

### Artefactos relevantes

* `Problema_1.ipynb`: análisis exploratorio, entrenamiento y evaluación.
* `mejor_modelo.keras`: modelo entrenado.

---

## Problema 2 — Reconocimiento de gestos (piedra, papel o tijera)

**Objetivo:** construir un pipeline completo de captura, entrenamiento y predicción en tiempo real de gestos manuales.

### Componentes

* **Captura de datos:** `record-dataset.py` (usa webcam para crear `rps_dataset.npy`).
* **Entrenamiento:** `train-gesture-classifier.py` (genera `rps_model.h5`).
* **Inferencia en tiempo real:** `rock-paper-scissors.py` (clasifica gestos usando MediaPipe + modelo entrenado).

### Uso recomendado para el usuario final

Ejecutar únicamente:

```bash
python rock-paper-scissors.py
```

con la cámara disponible y el archivo `rps_model.h5` en el mismo directorio.

---

## Problema 3 — Clasificación de escenas naturales con CNN

**Objetivo:** clasificar imágenes en seis categorías:

* `buildings`
* `forest`
* `glacier`
* `mountain`
* `sea`
* `street`

### Dataset

* Aproximadamente 25.000 imágenes de tamaño 150×150 distribuidas en carpetas de entrenamiento, test y predicción.
* Referencia: enlace de Google Drive documentado dentro del notebook.

### Artefactos relevantes

* `TP1_AAII_1C_2025.ipynb`: preparación de datos, entrenamiento y evaluación del modelo CNN.
* Carpetas `train/`, `test/`, `pred/` con la estructura esperada por el pipeline.

---

## Cómo ejecutar

### 1. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.venv\\Scripts\\activate  # Windows

pip install -r requirements.txt
```

Si no existe `requirements.txt`, instalar manualmente los paquetes necesarios según cada problema.

---

### 2. Ejecutar según el problema

* **Problema 1 y Problema 3:** abrir los notebooks con Jupyter Notebook o JupyterLab.

```bash
jupyter notebook
```

* **Problema 2:** ejecutar el script de inferencia en tiempo real.

```bash
python rock-paper-scissors.py
```
