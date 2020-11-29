import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
import sklearn
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import numpy as np
import pickle
import dlib
import cv2
import os
import time

def predict_face(yhat1, model1, **kwargs):
    in_encoder = Normalizer(norm = 'l2')
    yhat1 = in_encoder.transform(yhat1)
    results = model1.predict(yhat1)
    label = results[0]
    proba = model1.predict_proba(yhat1)
    probability = round(proba[0][label], 2)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    predicted_class_label = out_encoder.inverse_transform(results)
    label = predicted_class_label[0]
    return label, probability


if __name__=='__main__':

    data = np.load('./model/get_embedding_faces.npz', allow_pickle = True)
    trainX, testX, trainy, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    filename = './model/facenet_svm.sav'
    model = load_model('./model/facenet_keras.h5')
    hog_face_detector = dlib.get_frontal_face_detector()
    model1 = pickle.load(open(filename, 'rb'))
    print("[INFO] sampling frame from webcam ...")
    cap = cv2.VideoCapture(0)
    fps = FPS().start()
    while (True):
        _, frame = cap.read()
        faces_hog = hog_face_detector(frame, 1)
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            img = frame[x:x+w, y:y+h]
            img = cv2.resize(img, (160, 160))
            img = img.astype('float32')
            mean, std = img.mean(), img.std()
            img = (img - mean)/std
            img = np.expand_dims(img, axis=0)
            yhat = model.predict(img)
            yhat1 = yhat.reshape((1, -1))
            label, probal = predict_face(yhat1, model1)
            org = (x + 5, y + h + 9)
            if (probal >= 0.1):
                cv2.rectangle(frame, (x, y),(x + w, y + h), (0, 155, 255), 2)
                cv2.putText(frame, label + ':' + str(probal), org, fontScale=0.4,
                            color=(0, 0, 255), thickness=1, fontFace=4, lineType=1)
            fps.update()
        cv2.resize(frame, (640, 480))
        fps.stop()
        print("[INFO] FPS:{:.2f}".format(fps.fps()))
        cv2.imshow('Face_Reco', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()










