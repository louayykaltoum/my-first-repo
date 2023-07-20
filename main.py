import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import simpledialog
from deepface import DeepFace
import threading
import os
import json


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

c = 0
save_dir = "saved_faces"
os.makedirs(save_dir, exist_ok=True)
data_file = "data.json"

if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
    with open(data_file, 'r') as f:
        data = json.load(f)
else:
    data = {}  


for filename in os.listdir(save_dir):
    name = os.path.splitext(filename)[0] 
    image_path = os.path.join(save_dir, filename)
    if name not in data:
        data[name] = {'image_path': image_path, 'present': 0}

def save_face(frame, name):
    image_path = os.path.join(save_dir, f"{name}.jpg")
    cv.imwrite(image_path, frame)
    print(f"Face of '{name}' saved to {image_path}.")

    data[name] = {'image_path': image_path, 'present': 0}
    save_data_to_json()

def save_data_to_json():
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=4)

def check_face(frame):
    match_name = None
    for filename in os.listdir(save_dir):
        image_path = os.path.join(save_dir, filename)
        saved_face = cv.imread(image_path)
        try:
            if DeepFace.verify(frame, saved_face)['verified']:
                match_name = os.path.splitext(filename)[0]  
                break
        except ValueError as e:
            print("Error:", e)
    return match_name

def get_name_from_user():
    root = tk.Tk()
    root.withdraw() 
    name = simpledialog.askstring("Enter Your Name", "Please enter your name:")
    root.destroy()
    return name

def save_image_popup(frame):
    name = get_name_from_user()
    if name:
        save_face(frame, name)
        tk.messagebox.showinfo("Saved", f"Face of '{name}' saved successfully!")

while True:
    ret, frame = cap.read()
    if ret:
        if c % 30 == 0:
            matched_name = check_face(frame.copy())
            if matched_name:
                cv.putText(frame, f"COMPATIBLE: {matched_name}", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                data[matched_name]['present'] = 1
                save_data_to_json()
            else:
                cv.putText(frame, "NO MATCH", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        c += 1
        cv.imshow("video", frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_image_popup(frame)

cv.destroyAllWindows()
