from tensorflow.keras.models import load_model
import numpy as np
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import cv2
import os
import pickle
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
    model1 = pickle.load(open(filename, 'rb'))
    detector = MTCNN()
    print("[INFO] Sampling frame from webcam ...")
    fps = FPS().start()
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('./Output/' + 'quan.avi', fourcc, 15.0, size)
    while (True):
        _, frame = cap.read()
        start = time.time()
        results = detector.detect_faces(frame)
        end = time.time()
        print('Time detect face in a frame.{}'.format(end - start))
        if results != []:
            for person in results:
                bounding_box = person['box']
                img1 = frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                       bounding_box[0]:bounding_box[0] + bounding_box[2]]
                if img1.size != 0:
                    img = cv2.resize(img1, (160, 160))
                    img = img.astype('float32')
                    mean, std = img.mean(), img.std()
                    img = (img - mean) / std
                    img = np.expand_dims(img, axis=0)
                    yhat = model.predict(img)
                    yhat1 = yhat.reshape((1, -1))
                    label, probal = predict_face(yhat1, model1)
                    org = (bounding_box[0]+5, bounding_box[1] + bounding_box[3] + 9)
                    if (probal >= 0.6):
                        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                      (0, 155, 255), 2)
                    cv2.putText(frame, label + ':' + str(probal), org, fontScale=0.4, color = (0, 0, 255),
                                thickness=2, fontFace=4, lineType=2)
                    #font = cv2.FONT_HERSHEY_SIMPLEX
                    fps.update()
        cv2.resize(frame, (640, 480))
        out.write(frame)
        fps.stop()
        print("[INFO] FPS:{:.2f}".format(fps.fps()))
        cv2.imshow('Face_Reco', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
