import tkinter as tk
import threading
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
import chime

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
        background_image = Image.open("img/disCam3.png")  # Reemplaza con la ruta de tu imagen de fondo
        background_photo = ImageTk.PhotoImage(background_image)

        background_label = tk.Label(root, image=background_photo)
        background_label.image = background_photo
        background_label.place(relwidth=1, relheight=1)

        # Label para mostrar el frame de la CAM1
        self.cam1_label = tk.Label(root)
        self.cam1_label.place(x=13, y=29)  # Ajusta las coordenadas según tus necesidades

        # Label para mostrar el frame de la CAM 2
        self.cam2_label = tk.Label(root)
        self.cam2_label.place(x=508, y=29)  # Ajusta las coordenadas según tus necesidades

        # Botón para iniciar la cámara CAM 1
        self.active_cam1_button = tk.Button(root, text="Activar CAM1", command=self.active_cam1)
        self.active_cam1_button.place(x=1080, y=175)

        # Botón para iniciar la cámara CAM2
        self.active_cam2_button = tk.Button(root, text="Activar CAM2", command=self.active_cam2)
        self.active_cam2_button.place(x=1080, y=210)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='model/best.pt')

        # Label para mostrar la imagen de alerta
        self.alert_label = tk.Label(root)
        self.alert_label.place(x=1023, y=390)  # Posición de la alerta

        self.cap_cam1 = None
        self.cap_cam2 = None
        self.thread_cam1 = None
        self.thread_cam2 = None
        self.is_running_cam1 = False
        self.is_running_cam2 = False

    def active_cam1(self):
        self.cap_cam1 = cv2.VideoCapture(1)
        if not self.cap_cam1.isOpened():
            print("No se puede abrir la cámara USB")
            return

        self.is_running_cam1 = True
        self.thread_cam1 = threading.Thread(target=self.detect_objects_cam1)
        self.thread_cam1.start()

    def active_cam2(self):
        self.cap_cam2 = cv2.VideoCapture(2)
        if not self.cap_cam2.isOpened():
            print("No se puede abrir la cámara RTSP")
            return

        self.is_running_cam2 = True
        self.thread_cam2 = threading.Thread(target=self.detect_objects_cam2)
        self.thread_cam2.start()

    def detect_objects_cam1(self):
        while self.is_running_cam1:
            ret, frame = self.cap_cam1.read()

            # Cambia el tamaño a 200x200
            resized_frame = cv2.resize(frame, (485, 331))
            # Realizamos la detección
            detect = self.model(resized_frame)

            info_cam1 = detect.pandas().xyxy[0]  # Obtenemos la información de la detección
            print(info_cam1)  # Mostramos la información

            if info_cam1['confidence'].max() >= 0.92:
                chime.theme('pokemon')
                chime.error(sync=True)

                # Muestra la imagen de alerta
                alert_image_cam1 = Image.open("img/alerCam1.png")  # Reemplaza con la ruta de tu imagen de alerta
                alert_photo = ImageTk.PhotoImage(alert_image_cam1)
                self.alert_label.config(image=alert_photo)
                self.alert_label.image = alert_photo

            # Mostramos FPS
            rendered_frame = np.squeeze(detect.render())
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rendered_frame)
            img = ImageTk.PhotoImage(img)

            self.cam1_label.img = img
            self.cam1_label.configure(image=img)
            self.cam1_label.update_idletasks()

    def detect_objects_cam2(self):
        while self.is_running_cam2:
            ret, frame = self.cap_cam2.read()
            resized_frame = cv2.resize(frame, (485, 331))
            # Realizamos la detección
            detect = self.model(resized_frame)

            info_cam2 = detect.pandas().xyxy[0]  # Obtenemos la información de la detección
            print(info_cam2)  # Mostramos la información

            if info_cam2['confidence'].max() >= 0.92:
                chime.theme('pokemon')
                chime.error(sync=True)

                # Muestra la imagen de alerta
                alert_image_cam2 = Image.open("img/alerCam2.png")  # Reemplaza con la ruta de tu imagen de alerta
                alert_photo = ImageTk.PhotoImage(alert_image_cam2)
                self.alert_label.config(image=alert_photo)
                self.alert_label.image = alert_photo

            # Mostramos FPS
            rendered_frame = np.squeeze(detect.render())
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rendered_frame)
            img = ImageTk.PhotoImage(img)

            self.cam2_label.img = img
            self.cam2_label.configure(image=img)
            self.cam2_label.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()
