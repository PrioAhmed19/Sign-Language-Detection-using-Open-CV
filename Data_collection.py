import os
import cv2

#give the folder location where the dataset will be stored
DATA_DIR = 'F:/Project/New folder/Sign Language Detection Using Machine Learning'

#if the location is not created , create one
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


#for this project we take 5 classes of data
classes = 5

#for each dataclass we take 100 pics
data_collected_for_each_class = 100


#0 indicates for camera index ; if 0 doesnt work try 1/2
cap= cv2.VideoCapture(0)




#create sub folder for each class

for j in range(classes):
    if not os.path.exists(os.path.join(DATA_DIR,str(j))):
        os.makedirs(os.path.join(DATA_DIR,str(j)))

        print("Collecting data for class{}.".format(j))

    done  = False
    while True:
        ret,frame = cap.read()

    # the 100,50 is the position ,customize it accordingly
    #1.3 is the font size
    # 0,255,0 color od the text
    # 3 text thickness
    #cv.Line_AA anti alisted line
        start_point = (150,100)
        end_point = (450,400)
        color = (0,255,0)
        thickness = 2
        cv2.rectangle(frame,start_point,end_point,color,thickness)
        cv2.putText(frame,f'Press V to take a snap for class{j} ',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    
    #display current frame
        cv2.imshow('frame',frame)

    #check key press for 25 ms, customize it accordingly and if v is pressed start capturing frame
        if cv2.waitKey(25) == ord('v'):
            break

    counter = 0 

    #looping till take 100 pics as declared
    while counter < data_collected_for_each_class:
    #capturing single frame
        ret,frame = cap.read()

    
        cv2.imshow('frame',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR,str(j),'{}.jpg'.format(counter)),frame)
        counter +=1

    folder_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if not ret:
        print("Failed to capture frame")



#closing program
cap.release()
cv2.destroyAllWindows()




