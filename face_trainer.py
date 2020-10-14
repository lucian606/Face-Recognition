import cv2
import numpy as np
from PIL import Image
import os

#gets the images and label data
def getImagesAndLabels(path):
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') #converts to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        faces = face_cascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y: y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
print('[INFO] {0} faces trained.'.format(len(np.unique(ids))))
