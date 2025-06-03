import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Inicializar la webcam
cap = cv2.VideoCapture(1)

# Número de fotogramas para suavizado
num_frames = 15  
L_history, a_history, b_history = [], [], []

# Nombre del archivo CSV
data_file = "color_measurements.csv"

# Verificar si el archivo existe para agregar encabezados solo una vez
file_exists = os.path.isfile(data_file)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break

    # Definir área de interés (ROI)
    height, width, _ = frame.shape
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4

    # Dibujar el cuadro en pantalla
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Recortar el ROI y convertir a CIELAB
    roi_lab = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(roi_lab)

    # **Corrección del escalado**
    L_scaled = L * (100 / 255)   # De 0-255 a 0-100
    a_scaled = a - 128           # De 0-255 a -128 a 127
    b_scaled = b - 128           # De 0-255 a -128 a 127

    # Calcular los valores promedio
    L_mean, a_mean, b_mean = np.mean(L_scaled), np.mean(a_scaled), np.mean(b_scaled)

    # Aplicar suavizado (media móvil)
    L_history.append(L_mean)
    a_history.append(a_mean)
    b_history.append(b_mean)

    if len(L_history) > num_frames:
        L_history.pop(0)
        a_history.pop(0)
        b_history.pop(0)

    L_smooth, a_smooth, b_smooth = np.mean(L_history), np.mean(a_history), np.mean(b_history)

    # Mostrar en pantalla
    cv2.putText(frame, f"L*: {L_smooth:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"a*: {a_smooth:.2f}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"b*: {b_smooth:.2f}", (x1, y1 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Webcam con valores CIELAB", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Guardar datos en CSV al presionar 'f'
    if key == ord('f'):
        with open(data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Fecha y Hora", "L*", "a*", "b*"])
                file_exists = True
            
            # Obtener fecha y hora actual
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            writer.writerow([timestamp, L_smooth, a_smooth, b_smooth])
            print(f"Medición guardada - {timestamp} - L*: {L_smooth:.2f}, a*: {a_smooth:.2f}, b*: {b_smooth:.2f}")
    
    # Salir con 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()