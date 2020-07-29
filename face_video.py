import cv2
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import numpy as np
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import argparse
import time
parse=argparse.ArgumentParser()
parse.add_argument("path")
args=parse.parse_args()
gender_label=['Fermale','Male']
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
with open('emotion.json','r') as json_file:
	model2=model_from_json(json_file.read())
model2.load_weights("emotion.h5")
with open('gender.json', 'r') as json_file1:
	model3=model_from_json(json_file1.read())
model3.load_weights("gender1.h5")
data = np.load('get_embedding_faces.npz')
trainX, testX, trainy, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
filename='facenet_svm.sav'
model = load_model('facenet_keras.h5')
model1 = pickle.load(open(filename, 'rb'))
detector=MTCNN()
prevTime = 0
font=cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.4
color1=(0,0, 255)
thickness=1
def plot_icon(img, label, box):
	s_img=cv2.imread(os.path.join('./emoij_icon',label+'.png'))
	s_img=cv2.resize(s_img,(35, 35))
	x_offset=box[0]
	y_offset=box[1]-35
	y1, y2 = y_offset, y_offset + s_img.shape[0]
	x1, x2 = x_offset, x_offset + s_img.shape[1]
	alpha_s = s_img[:, :, 2] / 255.0
	alpha_l = 1.0 - alpha_s
	for c in range(0, 3):
    		img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
	return img
def plot_icon_gender(img, label, box):
	s_img=cv2.imread(os.path.join('./emoij_icon',label+'.png'))
	s_img=cv2.resize(s_img,(35, 35))
	x_offset=box[0]+box[2]
	y_offset=box[1]+box[3]-35
	y1, y2 = y_offset, y_offset + s_img.shape[0]
	x1, x2 = x_offset, x_offset + s_img.shape[1]
	alpha_s = s_img[:, :, 2] / 255.0
	alpha_l = 1.0 - alpha_s
	for c in range(0, 3):
    		img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
	return img
def convert_image(img):
	rgb_weights = [0.2989, 0.5870, 0.1140]
	grayscale_image = np.dot(img[...,:3], rgb_weights)
	return grayscale_image
def predict_gender(img, model3):
	result=model3.predict(img)
	score=np.amax(result)
	number=np.where(result==np.amax(result))
	label=gender_label[number[0][0]]
	return str(score), label
def predict_emotion(img, model2):
	result=model2.predict(img)
	score=np.amax(result)
	number=np.where(result==np.amax(result))
	if(len(number)!=1):
		label1=emotion_labels[int(number[1][0])]
	else:
		label1=emotion_labels[int(number[0][0])]
	return score, label1
def predict_face(yhat1, model1):
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
cap=cv2.VideoCapture(args.path)
filename=args.path.split('.')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('./Output/'+filename[0]+'.avi', fourcc, 15.0, size)
while(True):
	ret, frame=cap.read()
	curTime = time.time()
	sec = curTime - prevTime
	prevTime = curTime
	fps = 1/(sec)
	if ret == True:
		results=detector.detect_faces(frame)
		if results != []:
			for person in results:
				bounding_box=person['box']
				cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),(0, 155, 255), 2)
				img1=frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
				if img1.size !=0:
					img=cv2.resize(img1,(160, 160))
					img2=convert_image(img)
					img2=cv2.resize(img2, (48, 48)).flatten().reshape(1,48,48,1)
					img3=cv2.resize(img1,(80,80)).flatten().reshape(1, 80, 80,3)
					score2, label2=predict_gender(img3, model3)
					img=img.astype('float32')
					mean, std=img.mean(), img.std()
					img=(img-mean)/std
					img= np.expand_dims(img, axis=0)
					score, label1=predict_emotion(img2, model2)
					yhat=model.predict(img)
					yhat1=yhat.reshape((1,-1))
					label, probal=predict_face(yhat1, model1)
					#font=cv2.FONT_HERSHEY_SIMPLEX
					#fontScale = 0.4
					color=(255, 255, 255)
					#color1=(0,0, 255)
					color2=(255,255,0)
					#thickness=1
					org=(bounding_box[0], bounding_box[1]+bounding_box[3]+9)
					org1=(bounding_box[0], bounding_box[1]+bounding_box[3]+20)
					org2=(bounding_box[0], bounding_box[1]+bounding_box[3]+35)
					frame=plot_icon(frame, label1, bounding_box)
					frame=plot_icon_gender(frame, label2, bounding_box)
					cv2.putText(frame, label1+':'+str(score), org1, font, fontScale, color2, thickness, lineType=1)
					cv2.putText(frame, label2+':'+str(score), org2, font, fontScale, color2, thickness, lineType=1)
					if(probal >= 0.85):
						cv2.putText(frame, label+':'+str(probal), org, font, fontScale, color, thickness, lineType=1)
					else:
						label='unknown'
						cv2.putText(frame, label, org, font, fontScale, color, thickness, lineType=1)
	cv2.putText(frame, 'FPS:'+str(int(fps)),(10, 10), font, fontScale, color1, thickness, lineType=1)
	out.write(frame)
	cv2.imshow('Face_Reco',frame)
	if cv2.waitKey(1) &0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyAllWindows()
