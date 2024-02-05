# Import Librarys
import torch
import cv2
import numpy as np
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Read model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'C:/Users/ASUS VIVOBOOK\PycharmProjects/CustomYolov5/model/best.pt')

# Iniciamos VideoCapture
cap = cv2.VideoCapture(0)

# Loop
while True:
    # Leemos frame a frame
    ret, frame = cap.read()

    # Detectamos objetos
    results = model(frame)

    # Mostramos resultados
    cv2.imshow('Detector de Armas', np.squeeze(results.render()))

    # Si se presiona la tecla ESC se cierra el programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
