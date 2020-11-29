import tensorflow as tf
import tensorflow
from numpy import np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_devices = tensorflow.experimental.list_physical_devices()
tensorflow.experimental.set_memory_growth(gpu_devices[0], True)
gpus = tensorflow.test.gpu_device_name()
print('GPU:'+gpus)

def read_data(path, **kwargs):
    X, Y = []
    for file in os.listdir(path):
        for file_name in os.listdir(os.path.join(path, file)):
            img = cv2.imread(os.path.join(path, file, file_name))
            img = cv2.resize(img, (160, 160))
            X.append(img)
            Y.append(file)
    return np.asarray(X), np.asarray(Y)
def model(**kwargs):
    base_model = load_model('./model/facenet_keras.h5')
    for layer in base_model.layers[:-3]:
        layer.trainable = False

    return base_model

if __name__=='__main__':

    model = model()
    X, Y = read_data('./face1')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("[INFO] Example for traing: {:.2f}".format(len(y_train)))
    print("[INFO] Example for traing: {:.2f}".format(len(y_test)))
    x_train /= 255
    x_test /= 255
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    '''
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=8)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=8)
    '''
    callbacks = [tf.keras.callbacks.Callback.EarlyStopping(monitor='val_acc', patience=5),
                 tf.keras.callbacks.Callback.ModelCheckpoint('model_checkpoint',
                                                             monitor='val_acc',
                                                             verbose=0,
                                                             save_best_only=True,
                                                             save_weights_only=True),
                 tf.keras.callbacks.Callback.TensorBoard(log_dir='logs'),
                 tf.keras.callbacks.Callback.ReduceLROnPlateau(monitor='val_acc',
                                                               factor=0.1,
                                                               patience=5,
                                                               verbose=0)]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=tfa.keras.losses.TripletSemiHardLoss(),
                  metrics=['acc'])

    model.fit(x_train, y_train, batch_size=32, steps_per_epoch=len(x_train)/32, epochs=20,
              validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)

    model.save('/model/facenet.h5')

    print('Accuracy: {:.2f}'.format(model.evaluate(x_test, y_test)))







