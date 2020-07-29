import cv2
import os
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import argparse

#parse=argparse.ArgumentParser()
#parse.add_argument("path")
#args=parse.parse_args()

data = np.load('get_embedding_faces.npz')
trainX, testX, trainy, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
filename='facenet_svm.sav'
model = load_model('facenet_keras.h5')
model1 = pickle.load(open(filename, 'rb'))
detector=MTCNN()
def fileuio(yhat1, model1):
	in_encoder = Normalizer(norm='l2')
	yhat1=in_encoder.transform(yhat1)
	results=model1.predict(yhat1)
	label=results[0]
	proba=model1.predict_proba(yhat1)
	probability=round(proba[0][label], 2)
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	predicted_class_label=out_encoder.inverse_transform(results)
	label=predicted_class_label[0]
	return label, probability
cap=cv2.VideoCapture(0)
while(True):
	ret, frame=cap.read()
	results=detector.detect_faces(frame)
	if results != []:
		for person in results:
			bounding_box=person['box']
			cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),(0, 155, 255), 2)
			img1=frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
			if len(img1) != 0:
				img1=cv2.resize(img1,(160, 160))
				img1=img1.astype('float32')
				mean, std=img1.mean(), img1.std()
				img1=(img1-mean)/std
				img1= np.expand_dims(img1, axis=0)
				yhat=model.predict(img1)
				yhat1=yhat.reshape((1,-1))
				label, probal=fileuio(yhat1, model1)
				font=cv2.FONT_HERSHEY_SIMPLEX
				fontScale = 0.4
				color=(0, 0, 255)
				thickness=1
				org=(bounding_box[0]+9, bounding_box[1]+bounding_box[3]+9)
				if(probal >=0.5):
					cv2.putText(frame, label+':'+str(probal), org, font, fontScale, color, thickness, lineType=2)
				else:
					label='None'
					cv2.putText(frame, label, org, font, fontScale, color, thickness, lineType=2)
				cv2.imshow('Face_Reco',frame)
			if cv2.waitKey(1) &0xFF == ord('q'):
				break
cap.release()
cv2.destroyAllWindows()
