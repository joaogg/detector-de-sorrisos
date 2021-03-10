import cv2
import numpy as np
import sys
import matplotlib.pylab as plt
from datetime import datetime
import time

facePath = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"
smilePath = "C:\opencv\sources\data\haarcascades\haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(0)
cap.set(3,854)
cap.set(4,480)

sF = 1.05

wq = 0
tirarFoto = 0
verificaSorriso = False;
#img2 = cv2.imread('cc.png')

while wq==0:

    now = datetime.now()
    ret, frame = cap.read() # Capture frame-by-frame
    ret, sera = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # ---- Draw a rectangle around the faces

    rostosImagem = 0
    sorrisoImagem = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (35,142,35), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        rostosImagem = rostosImagem + 1
        #cv2.imshow("ROI", roi_color)

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        verificaSorriso = False

    # Set region of interest for smiles
        for (x, y, w, h) in smile:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0,0,156), 2)
            verificaSorriso = 0
            if len(smile)>0:
                sorrisoImagem = sorrisoImagem + 1
                tirarFoto = tirarFoto + 1

                verificaSorriso = True
                #print ("Valor Variavel: " + str(tirarFoto))
                if tirarFoto>19:
                    #plt.imsave("ImgAAA.png", frame )
                    cv2.putText(sera,str(now.day) + str(now.month) + str(now.year) + str(now.hour) + str(now.minute) + str(now.second),(600,470), cv2.FONT_HERSHEY_SIMPLEX, 1,(5,51,255),2,cv2.LINE_AA)
                    cv2.imwrite('C:/Users/JoaoGabriel/Desktop/Projeto/fotos/'+str(now.day) + str(now.month) + str(now.year) + str(now.hour) + str(now.minute) + str(now.second)+'.png',sera)
                    print(" FOTO RETIRADA ")
                    tirarFoto=0

        if rostosImagem != sorrisoImagem:
            tirarFoto = 0

    #cv2.cv.Flip(frame, None, 1)
    #cv2.putText(frame,'SORRISOS: '+ str(tirarFoto),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)

    cv2.putText(frame,'SORRISO: '+str(verificaSorriso),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'ROSTOS NA IMAGEM: '+ str(rostosImagem),(10,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'SORRISOS NA IMAGEM: '+ str(sorrisoImagem),(10,130), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('Prototipo de Deteccao de Sorrisos para Selfies', frame)
    c = cv2.waitKey(7)  % 0x100
    if c == 1:
        break

cap.release()
cv2.destroyAllWindows()
