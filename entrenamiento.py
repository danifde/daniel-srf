import os
import cv2
import dlib
import numpy as np
import csv

# Rutas de datos y modelos
dataPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/data'
dlibModelPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/shape_predictor_68_face_landmarks.dat'
openfaceModelPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/nn4.small2.v1.t7'

# Inicializar el modelo dlib para la detección de rostros
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(dlibModelPath)

# Cargar el modelo OpenFace
net = cv2.dnn.readNet(openfaceModelPath)

# Preparar datos de entrenamiento
peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)

labels = []
embeddingsData = []

label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imágenes de', nameDir)

    for fileName in os.listdir(personPath):
        print('Rostro', nameDir + '/' + fileName)
        labels.append(label)

        # Leer la imagen
        image_path = os.path.join(personPath, fileName)
        bgrImg = cv2.imread(image_path)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY)

        # Detectar el rostro en la imagen
        faces = detector(gray)

        if len(faces) == 1:
            face = faces[0]
            shape = sp(gray, face)

            # Extraer embedding
            faceBlob = cv2.dnn.blobFromImage(bgrImg, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(faceBlob)
            rep = net.forward()

            embeddingsData.append(rep.flatten())

    label += 1

# Guardar los vectores de embeddings en un archivo CSV
embeddingsCSVPath = 'embeddings.csv'
with open(embeddingsCSVPath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(embeddingsData)

print("Vectores de embeddings guardados en:", embeddingsCSVPath)









# import cv2
# import os
# import numpy as np

# dataPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/data'
# peopleList = os.listdir(dataPath)
# print('lista de personas:', peopleList)

# labels = []
# facesData = []
# label = 0

# for nameDir in peopleList:
#     personPath = dataPath + '/' + nameDir
#     print('leyendo las imagenes')

#     for fileName in os.listdir(personPath):
#         print('rostros', nameDir + '/' + fileName)
#         labels.append(label)
#         facesData.append(cv2.imread(personPath + '/' + fileName, cv2.IMREAD_GRAYSCALE))

#     label = label + 1

# # Crear el reconocedor EigenFace
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# #face_recognizer = cv2.face.FisherFaceRecognizer_create()
# #face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# print("entrenando...")
# face_recognizer.train(facesData, np.array(labels))

# # Guardar el modelo entrenado
# face_recognizer.save('modeloeigenface.xml')
# #face_recognizer.save('modelofisherface.xml')
# #face_recognizer.save('modelolbphfface.xml')
# print("modelo entrenado")
