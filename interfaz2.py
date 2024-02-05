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
        self.root.title("Sistema de Monitoreo VisioSafe v1.0")

        # Establecer el tamaño de la ventana
        self.root.geometry("1280x720")

        # Agregar un fondo a la ventana
        background_image = Image.open("img/disCam.png")  # Reemplaza con la ruta de tu imagen de fondo
        background_photo = ImageTk.PhotoImage(background_image)

        background_label = tk.Label(root, image=background_photo)
        background_label.image = background_photo
        background_label.place(relwidth=1, relheight=1)

        # Label para mostrar el frame de la cámara USB
        self.video_label_usb = tk.Label(root)
        self.video_label_usb.place(x=13, y=29)  # Ajusta las coordenadas según tus necesidades

        # Label para mostrar el frame de la cámara IP
        self.video_label_ip = tk.Label(root)
        self.video_label_ip.place(x=508, y=29)  # Ajusta las coordenadas según tus necesidades

        # Botón para iniciar la cámara USB
        fondo_button = "#6AA84F"
        self.start_usb_button = ttk.Button(root, text="Activar CAM1", command=self.start_usb_video)
        self.start_usb_button.place(x=1077, y=135)

        # Botón para iniciar la cámara IP
        self.start_ip_button = ttk.Button(root, text="Activar CAM2", command=self.start_ip_video)
        self.start_ip_button.place(x=1077, y=220)

        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='C:/Users/ASUS VIVOBOOK/PycharmProjects/VisionSafev5/model/best.pt')

        self.cap_usb = None
        self.cap_ip = None
        self.thread_usb = None
        self.thread_ip = None
        self.is_running_usb = False
        self.is_running_ip = False

    def start_usb_video(self):
        self.cap_usb = cv2.VideoCapture(1)
        if not self.cap_usb.isOpened():
            print("No se puede abrir la cámara USB")
            return

        self.is_running_usb = True
        self.thread_usb = threading.Thread(target=self.detect_objects_usb)
        self.thread_usb.start()

    def start_ip_video(self):
        #url_rtsp = 'http://192.168.100.57:4747/video'  # Reemplaza con la URL correcta de tu cámara IP
        self.cap_ip = cv2.VideoCapture(2)
        if not self.cap_ip.isOpened():
            print("No se puede abrir la cámara RTSP")
            return

        self.is_running_ip = True
        self.thread_ip = threading.Thread(target=self.detect_objects_ip)
        self.thread_ip.start()

    def detect_objects_usb(self):
        while self.is_running_usb:
            ret, frame = self.cap_usb.read()

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

            self.video_label_usb.img = img
            self.video_label_usb.configure(image=img)
            self.video_label_usb.update_idletasks()

    def detect_objects_ip(self):
        while self.is_running_ip:
            ret, frame = self.cap_ip.read()
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

            self.video_label_ip.img = img
            self.video_label_ip.configure(image=img)
            self.video_label_ip.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()
