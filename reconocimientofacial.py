import cv2
import os
import dlib

dataPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/data'
imagePath = os.listdir(dataPath)
print('imagesPaths=', imagePath)

face_recognizer = cv2.face_EigenFaceRecognizer.create()
face_recognizer.read('modeloeigenface.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar el detector de rostros de Dlib y el predictor de puntos de referencia faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cap.read()
    if ret is False:
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
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            # ... Resto de tu código ...
        else:
            print('Recorte de cara fallido')

        cv2.putText(frame, '{}'.format(result), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 4000:
            cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'DESCONOCIDO', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
