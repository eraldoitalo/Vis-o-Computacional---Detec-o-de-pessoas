import cv2 as cv
import numpy as np

# Classificadores HaarCascade
pessoa = cv.CascadeClassifier('Classificadores/haarcascade_fullbody.xml') # Corpo inteiro
frontal_face = cv.CascadeClassifier('Classificadores/haarcascade_frontalface_default.xml') # Rosto

cap = cv.VideoCapture(0)

# Virificando
if not cap.isOpened():
    print("Não foi possivel abrir a webcam")
    exit()

# Loop para capturar e mostrar os frames 
while True:
    ret, frame = cap.read() # Ler frame por frame
    if not ret:
        print("Não foi possivel ler o frame")
        break
    
    # Converter a escala de cor para cinza (computador processa melhor)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Deteccao dos classificadores
    person = pessoa.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,)
    faces = frontal_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,)

    for (x, y, w, h) in person:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow('Reconhecimento de pessoas', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.realease()
cap.destroyAllWindows()