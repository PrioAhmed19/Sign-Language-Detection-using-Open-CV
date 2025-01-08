import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

DATA_DIR = 'F:/Project/New folder/Sign Language Detection Using Machine Learning/Dataset'
#using mediapipes hand module to detect hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles  #predefined styles for hand

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.5)  # checking its a pic not a video, minimum confidence threshhold 50% to detect as hand

#creating empty list for hand data and labels for each image
data = []
labels = []

#dataset processing using loop

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    print(f"Processing directory: {dir_path}")
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory: {dir_path}")
        continue

    for img_path in os.listdir(dir_path):
        full_path = os.path.join(dir_path, img_path)
        print(f"Processing file: {full_path}")

        # Filter for valid image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        if not img_path.lower().endswith(valid_extensions):
            print(f"Skipping non-image file: {img_path}")
            continue

        try:
            img = cv2.imread(full_path)
            if img is None:
                print(f"Failed to load image: {full_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                data.append(data_aux)
                labels.append(dir_)
        except Exception as e:
            print(f"Error processing file {full_path}: {e}")
            continue


f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()


import os
print("Saving pickle file in:", os.getcwd())
