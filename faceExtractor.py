# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:07:21 2020

@author: Biswadeep
"""
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
detector=MTCNN()
filenames = []
#faces=[]
#path='E:/MachineL/VGGFACE/database/realPics/'
count=0

#faces=detector.detect_faces(img)
img=cv2.imread('E:/MachineL/VGGFACE/database/realPics/IMG_20200211_143800.jpg')
img=cv2.resize(img,(500,500))
faces=detector.detect_faces(img)

for face in faces:
    x,y,w,h=face['box']
    crop_face = img[y:y+h, x:x+w]
    if face['confidence'] >=0.9:
        count+=1
        face = cv2.resize(crop_face,(300,300))
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'E:/MachineL/VGGFACE/lulu/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==27 or count==100:
        break
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')