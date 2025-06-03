import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Cargar el modelo entrenado
modelo = load_model("clasificador_ovas.h5")

# Clases del dataset (deben coincidir con el orden de tus carpetas en el dataset)
clases = ["Muerta", "Viva", "Cíclope"]

# Inicializar la cámara
camera = cv2.VideoCapture(0)  # Cambia a 1 o el índice correcto si tienes varias cámaras

if not camera.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar un frame de la cámara
    ret, frame = camera.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Preprocesar la imagen para que sea compatible con el modelo
    img = cv2.resize(frame, (128, 128))  # Redimensionar la imagen al tamaño que espera el modelo
    img = image.img_to_array(img)  # Convertir la imagen a un array NumPy
    img = np.expand_dims(img, axis=0) / 255.0  # Normalizar y expandir dimensiones para que sea compatible con el modelo

    # Hacer la predicción usando el modelo entrenado
    prediccion = modelo.predict(img)

    # Obtener la clase con la probabilidad más alta
    indice_predicho = np.argmax(prediccion)  # Devuelve el índice de la clase con mayor probabilidad
    clase_predicha = clases[indice_predicho]  # Obtener el nombre de la clase correspondiente

    # Mostrar la predicción en la imagen capturada
    cv2.putText(frame, f"Ova: {clase_predicha}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen con la predicción en una ventana
    cv2.imshow("Clasificación de Ovas", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
camera.release()
cv2.destroyAllWindows()