import cv2
import numpy as np
import os
import argparse
from mtcnn import MTCNN

detector=MTCNN()
path='./video'
a=0
for file in os.listdir(path):
    filename=file.split('.')
    dir='./face/'+filename[0]
    os.makedirs(dir)
    cap=cv2.VideoCapture(os.path.join(path, file))
    while(True):
        _, frame=cap.read()
        results=detector.detect_faces(frame)
        if results != []:
            for person in results:
                box=person['box']
                img=frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                if img.size != 0:
                    img=cv2.resize(img, (160,160))
                    cv2.imwrite(dir+'/'+str(a)+'.jpg', img)
                    a+=1
