import cv2
import os
import numpy as np

dataPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/data'
peopleList = os.listdir(dataPath)
print('lista de personas:', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('rostros', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, cv2.IMREAD_GRAYSCALE))

    label = label + 1

# Crear el reconocedor EigenFace
face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()


print("entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
face_recognizer.save('modeloeigenface.xml')
#face_recognizer.save('modelofisherface.xml')
#face_recognizer.save('modelolbphfface.xml')
print("modelo entrenado")
