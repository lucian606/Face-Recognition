import os
import cv2

#Python script which gathers a user's face and puts it into the data set
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

user = input('Enter your id:')
pictures = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pictures += 1;
        cv2.imwrite('dataset/' + user + '.' + str(pictures) + '.jpg', gray[y:y + h, x:x + w])
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif pictures >= 20:
        break
print("[INFO] Done creating the user's dataset")
cam.release()
cv2.destroyAllWindows()
