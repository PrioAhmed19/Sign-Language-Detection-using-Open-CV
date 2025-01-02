import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

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
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux = [] #for storing normalized data

    x_= []
    y_= []

    img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))  #reading image as bgr
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #converting to rgb as mediapipe requires rgb


    results = hands.process(img_rgb)  #process to detect hand marks
    if results.multi_hand_landmakrs:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):  #loops each image to detect landmark(21 landmakrs in a hand)
                x= hand_landmarks.landmark[i].x
                y= hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            #looping again for normalization

            for i in range (len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x-min(x_))
                data_aux.append(y - min(y_))
            data.append(data_aux)        #adding processed landmark in the image
            labels.append(dir_)  #adding class label 


f = open('data.pickle','wb')  #store binary version of the processed data
pickle.dump({'data':data,'labels':labels},f)  # to load the data later for training
f.close()   # close the file