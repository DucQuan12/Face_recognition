import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
data = np.load('get_embedding_faces.npz')
trainX, testX, trainy, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
nsamples, nx, ny = trainX.shape
trainX1=trainX.reshape((nsamples,nx*ny))
nsamples, nx, ny = testX.shape
testX1=testX.reshape((nsamples,nx*ny))
in_encoder = Normalizer(norm='l2')
trainX1 = in_encoder.transform(trainX1)
testX1 = in_encoder.transform(testX1)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

model = SVC(kernel='linear', probability=True)
model.fit(trainX1, trainy)

yhat_train = model.predict(trainX1)
yhat_test = model.predict(testX1)

score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
pickle.dump(model, open('facenet_svm.sav','wb'))

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
