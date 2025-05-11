# Este programa utiliza MediaPipe y OpenCV para capturar gestos de manos en tiempo real (piedra, papel o tijera) 
# y almacenarlos en un dataset que luego se utilizará para entrenar un modelo de deep learning.
# --------------------------------------------------------------------------------------------------------------

# Librerías utilizadas  
import cv2
import mediapipe as mp
import numpy as np

# Configuración para silenciar mensajes de TensorFlow (INFO y WARNINGS)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lista para almacenar los datos
dataset = []

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Diccionario que mapea códigos numéricos a gestos (etiquetado)
gestos = {0: "piedra", 1: "papel", 2: "tijera"}

# Inicializar la detección de manos
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No se pudo acceder a la cámara.")
            break

        # Se convierte el frame de BGR (formato OpenCV) a RGB (formato MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen
        results = hands.process(frame_rgb)

        # Dibujar landmarks si se detectan manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar instrucciones
        cv2.putText(frame, "0: piedra, 1: papel, 2: tijera, q: salir", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Muestras: {len(dataset)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Encabezado
        cv2.imshow('Piedra, papel o tijera', frame)

        # Espera una entrada de teclado durante un tiempo determinado
	# Almacena el código numérico de la tecla presionada
        key = cv2.waitKey(5) & 0xFF

        # Si se presiona '0', '1' o '2'
        if key in [ord('0'), ord('1'), ord('2')]:    # ord('0') → 48 
            etiqueta = int(chr(key))

            # Si hay manos detectadas
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extraer coordenadas (21 landmarks x 2 coordenadas = 42 valores)
                    coordenadas = []
                    for landmark in hand_landmarks.landmark:
                        coordenadas.extend([landmark.x, landmark.y])   # MediaPipe devuelve x e y normalizados (0-1)
                    
                    if len(coordenadas) == 42:
                        dataset.append({
                            'etiqueta': etiqueta,        
                            'coordenadas': coordenadas
                        })
                        print(f"Gesto '{gestos[etiqueta]}' guardado - Total: {len(dataset)}")
                    else:
                        print(f"Error: Se esperaban 42 valores, se obtuvieron {len(coordenadas)}")  

        # Salir con 'q'
        if key == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Guardar dataset en formato numpy 
dataset_def = {
    'etiquetas': np.array([d['etiqueta'] for d in dataset]),            #array de las clases (0, 1, 2)
    'coordenadas': np.array([d['coordenadas'] for d in dataset])        #array de coordenadas x y 
}
np.save('rps_dataset.npy', dataset_def)
print(f"Dataset guardado en 'rps_dataset.npy' con {len(dataset)} muestras")