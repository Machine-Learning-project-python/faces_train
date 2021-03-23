import os
import cv2 as cv
import numpy as np


#creamos lista con los nombres de los artitas que vamos a reconocer
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

#creamos una variable con la dirección donde vamos a tener las imágenes
DIR = r'./Faces/train'


#cargamos el clasificador
haar_cascade = cv.CascadeClassifier('haar_face.xml')


#Creamos el train
features = []
labels = []


def create_train():
    for person in people:
        #creamos el path  y el etiqueta de cada persona
        path = os.path.join(DIR, person)
        label = people.index(person)
        #leemos cada imagen
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)

            if img_array is None:
                continue 
            #convertimos a gris las imágenes    
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            #detectamos las caras.
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            #las guardamos
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()


print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)
#Creamos el reconocedor
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Entrenamos el reconocedor con las objetos que tenemos
face_recognizer.train(features,labels)
#guardamos los archivos generados para poder cargarlos a la hora validar las imágenes
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)