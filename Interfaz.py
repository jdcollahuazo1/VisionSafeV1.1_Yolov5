import tkinter as tk
from tkinter import ttk
import threading
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class ObjectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Monitoreo VisioSafe v1.1")

        # Establecer el tamaño de la ventana
        self.root.geometry("1280x720")

        # Agregar un fondo a la ventana
        background_image = Image.open("img/disCam.png")  # Reemplaza con la ruta de tu imagen de fondo
        background_photo = ImageTk.PhotoImage(background_image)

        background_label = tk.Label(root, image=background_photo)
        background_label.image = background_photo
        background_label.place(relwidth=1, relheight=1)

        # Label para mostrar el frame de la cámara USB
        self.cam1_label = tk.Label(root)
        self.cam1_label.place(x=13, y=29)  # Ajusta las coordenadas según tus necesidades

        # Botón para iniciar la cámara USB
        fondo_button = "#6AA84F"
        self.active_cam1_button = ttk.Button(root, text="Iniciar CAM 1", command=self.active_cam1)
        self.active_cam1_button.place(x=1077, y=135)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='C:/Users/ASUS VIVOBOOK/PycharmProjects/CustomYolov5/model/best.pt')

        self.cap_cam1 = None
        self.thread_cam1 = None
        self.is_running_cam1 = False

    def active_cam1(self):
        self.cap_cam1 = cv2.VideoCapture(1)
        if not self.cap_cam1.isOpened():
            print("No se puede abrir la cámara USB")
            return

        self.is_running_cam1 = True
        self.thread_cam1 = threading.Thread(target=self.detect_objects_usb)
        self.thread_cam1.start()

    def detect_objects_usb(self):
        while self.is_running_cam1:
            ret, frame = self.cap_cam1.read()

            # Cambia el tamaño a 200x200
            resized_frame = cv2.resize(frame, (485, 331))
            # Realizamos la detección
            detect = self.model(resized_frame)

            info = detect.pandas().xyxy[0]  # Obtenemos la información de la detección
            print(info)  # Mostramos la información

            # Mostramos FPS
            rendered_frame = np.squeeze(detect.render())
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rendered_frame)
            img = ImageTk.PhotoImage(img)

            self.cam1_label.img = img
            self.cam1_label.configure(image=img)
            self.cam1_label.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()
