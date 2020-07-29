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

with open('gender.json', 'r') as json_file1:
	model3=model_from_json(json_file1.read())
model3.load_weights("gender1.h5")
