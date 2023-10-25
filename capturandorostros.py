import dlib
import os
import cv2
import imutils

personName = 'yaman'
dataPath = 'C:/Users/sena/Desktop/repositorios/reconocimiento-facial/data'
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargamos el modelo de detección de rostros de Dlib
detector = dlib.get_frontal_face_detector()

count = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectamos rostros en la imagen
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostros_{}.jpg'.format(count), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 800:
        break

cap.release()
cv2.destroyAllWindows()