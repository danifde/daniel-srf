import cv2
import os
import dlib
import numpy as np

# Cargar los vectores de embeddings desde el archivo CSV
embeddingsCSVPath = 'embeddings.csv'
embeddingsData = np.genfromtxt(embeddingsCSVPath, delimiter=',')

dataPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/data'
peopleList = os.listdir(dataPath)  # Lista de nombres de las personas que entrenaste

# Cargar el detector de rostros de Dlib y el predictor de puntos de referencia faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

openfaceModelPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/nn4.small2.v1.t7'
model = cv2.dnn.readNet(openfaceModelPath)

cap = cv2.VideoCapture(0)

# Definir un umbral para el reconocimiento (ajusta según sea necesario)
umbral = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Utilizar el predictor de puntos de referencia faciales de Dlib
        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x_landmark, y_landmark = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x_landmark, y_landmark), 1, (0, 0, 255), -1)

        rostro = gray[y:y + h, x:x + w]

        # Verificar si el recorte de la cara fue exitoso
        if rostro.shape[0] > 0 and rostro.shape[1] > 0:
            # Convertir la imagen de escala de grises en una imagen de tres canales
            rostro_bgr = cv2.cvtColor(rostro, cv2.COLOR_GRAY2BGR)
            rostro_bgr = cv2.resize(rostro_bgr, (96, 96), interpolation=cv2.INTER_CUBIC)

            # Calcular el embedding del rostro detectado
            blob = cv2.dnn.blobFromImage(rostro_bgr, 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            embedding = model.forward()

            # Realizar el reconocimiento facial basado en los embeddings
            min_distance = float('inf')
            min_distance_label = None

            for label, reference_embedding in enumerate(embeddingsData):
                distance = np.linalg.norm(embedding - reference_embedding)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_label = label

            if min_distance < umbral:
                recognized_person = peopleList[min_distance_label]
                cv2.putText(frame, recognized_person, (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'DESCONOCIDO', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            print('Recorte de cara fallido')

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



