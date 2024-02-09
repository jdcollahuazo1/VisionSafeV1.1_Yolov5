import tkinter as tk
from tkinter import ttk
import threading
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
import pathlib
import chime

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

        # Label para mostrar el frame de la CAM 1
        self.cam1_label = tk.Label(root)
        self.cam1_label.place(x=13, y=29)  # Ajusta las coordenadas según tus necesidades


        # Label para mostrar la imagen de alerta
        self.alert_label = tk.Label(root)
        self.alert_label.place(x=1023, y=390)  # Posición de la alerta

        # Botón para iniciar la CAM 1
        self.active_cam1_button = tk.Button(root, text=" Activar CAM 1", command=self.active_cam1)
        self.active_cam1_button.place(x=1080, y=175)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='model/best.pt')

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

            if info['confidence'].max() >= 0.92:
                chime.theme('pokemon')
                chime.error(sync=True)

                # Muestra la imagen de alerta
                alert_image = Image.open("img/alerCam1.png")  # Reemplaza con la ruta de tu imagen de alerta
                alert_photo = ImageTk.PhotoImage(alert_image)
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


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()
