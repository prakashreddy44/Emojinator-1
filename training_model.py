import pickle
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.set_image_dim_ordering('tf')

def training_model():
    model = Sequential()
    model.add(Conv2D(32, (4,4), input_shape=(110, 110, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(5, activation='softmax'))
    sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath="trained_model.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    return model,callbacks_list

def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
        
    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)
    train_images = np.reshape(train_images, (train_images.shape[0], 110, 110, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], 110, 110, 1))
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    train_labels=train_labels[:,1:]
    
    test_labels=test_labels[:,1:]
    model,callbacks_list = training_model()
    model.summary()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=16, batch_size=128,callbacks=callbacks_list)

	#model.save('cnn_model_keras2.h5')

train()
