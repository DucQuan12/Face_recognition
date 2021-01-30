from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import pickle

def read_face(path):
    X = []
    Y = []
    for filename in os.listdir(path):
        for file in os.listdir(os.path.join(path, filename)):
            img = cv2.imread(os.path.join(path, filename, file))
            img = cv2.resize(img, (160, 160))
            X.append(img)
            Y.append(filename)
    return np.asarray(X), np.asarray(Y)

def get_embedding(model, pixel, **kwargs):
    pixel = pixel.astype(np.float32)
    mean, std = pixel.mean(), pixel.std()
    pixel = (pixel - mean) / std
    sample = np.expand_dims(pixel, axis=0)
    yhat = model.predict(sample)

    return yhat


def save_array(X, Y):
    x, y, z, t = train_test_split(X, Y, test_size=0.2, random_state=220)
    np.savez_compressed('dataset_faces.npz', x, y, z, t)


def newsave_array(model, **kwargs):
    newx_train = list()
    newx_test = list()
    data = np.load('dataset_faces.npz')
    x_train, x_test, y_train, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    for pixel in x_train:
        yhat = get_embedding(model, pixel)
        newx_train.append(yhat)
    for pixel in x_test:
        yhat = get_embedding(model, pixel)
        newx_test.append(yhat)

    np.savez_compressed('get_embedding_faces.npz', newx_train, newx_test, y_train, y_test)


if __name__ == '__main__':
    model = load_model('facenet_keras.h5')
    X, Y = read_face('./face1')
    save_array(X, Y)
    newsave_array(model)
