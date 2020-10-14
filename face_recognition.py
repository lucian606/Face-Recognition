import numpy as np
import cv2
import os

#Loading the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
#Initializing the list of names (ids used as indexes)
names = ['None', 'Lucian']
#Loading the cascades needed for the face detection
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#Creating a screen capture and setting the width/height
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
#Defining min windows size to be recognized as a face
minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Finding the faces in our image
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH))) #The list of faces found using the cascade
    #Marking the faces with a rectangle with the corner at (x,y) with width = w and height = h
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
        #The recognizer will return the id of the picture and how much confidence is in relation with this match
        id, confidence = recognizer.predict(gray[y: y + h, x: x +w])
        if confidence < 100:
            name = names[id]
            confidence = '  {0}%'.format(round(100 - confidence))
        else:
            name = names[0]
            confidence = '   {0}%'.format(round(100 - confidence))
        cv2.putText(img, str(name), (x + 5, y + 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    cv2.imshow('camera',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
print('[INFO] Exiting Program.')
cap.release()
cv2.destroyAllWindows()
