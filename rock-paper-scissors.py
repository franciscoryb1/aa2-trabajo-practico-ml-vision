# Este programa utiliza un modelo pre-entrenado de deep learning junto con MediaPipe para reconocer gestos de mano 
# (piedra, papel o tijera) en tiempo real a través de la cámara web, mostrando las predicciones en pantalla con OpenCV.
# ---------------------------------------------------------------------------------------------------------------------

# Librerías utilizadas  
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Configuración del entorno de ejecución 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'           # Silenciar los mensajes de TensorFlow
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'   # Deshabilitar backend MSMF de OpenCV para usar DirectShow
os.environ['MEDIAPIPE_GPU'] = '0'                  # Forzar uso de CPU en MediaPipe

# Ajustes para estabilidad de OpenCV
cv2.ocl.setUseOpenCL(False)  # Deshabilitar OpenCL
cv2.setNumThreads(1)         # Usar solo un hilo

# Cargar recursos
try:
    model = load_model('rps_model.h5')
except Exception as e:
    print(f"Error al cargar recursos: {e}")
    exit()

# Configuración de MediaPipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
gestos = {0: "PIEDRA", 1: "PAPEL", 2: "TIJERA"}

# Configuración de cámara 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # Usar backend DirectShow
if not cap.isOpened():
    cap = cv2.VideoCapture(0)                 # Intentar con backend por defecto
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        exit()

# Reducir resolución para mejor rendimiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Configuración de detección de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Crear ventana 
cv2.namedWindow('Detección de Gestos', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detección de Gestos', 800, 600)

try:
    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Procesamiento de imagen
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Se convierte el frame de BGR a RGB
        results = hands.process(frame_rgb)                  # Procesa una imagen en busca de manos
       
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
               
                # Extraer coordenadas
                coordenadas = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
               
                if len(coordenadas) == 42:
                    try:
                        input_data = coordenadas.reshape(1, -1)
                        pred = model.predict(input_data, verbose=0)
                        gesto = gestos[np.argmax(pred)]
                       
                        # Mostrar resultado
                        cv2.putText(frame, gesto, (50, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    except Exception as e:
                        print(f"Error en predicción: {e}")

        # Encabezado 
        cv2.imshow('Detección de Gestos', frame)
       
        # Salir con 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    # Limpieza final
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    os.system('pkill -f "python"')