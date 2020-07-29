import numpy as np
import os
import sklearn
import tensorflow as tf
from tensorflow.keras.models import load_model
from processing_data import read_face
from processing_data import get_embedding
from processing_data import save_array
from processing_data import newsave_array
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN

model = load_model('facenet_keras.h5')

X, Y=read_face()
save_array(X, Y)
newsave_array(model)
