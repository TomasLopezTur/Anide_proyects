import cv2

# Inicializar la cámara
camera = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not camera.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar un frame
    ret, frame = camera.read()

    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Mostrar el frame en una ventana
    cv2.imshow("Vista previa", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
camera.release()
cv2.destroyAllWindows()