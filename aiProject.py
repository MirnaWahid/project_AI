# -*- coding: utf-8 -*-
import os
import librosa
import numpy as np
from keras.layers import Dropout, Flatten, Conv1D, Input, MaxPooling1D, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

train_audio_path = 'M:/AI_Project/16000_pcm_speeches'
labels = os.listdir(train_audio_path)
all_wave = []
all_label = []                   
for label in labels:              
    print(label)                  
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000) 
        # print(train_audio_path + '/' + label + '/' + wav)
        samples = librosa.resample(samples, sample_rate, 8000)
        print(len(samples))
        if (len(samples) == 8000) :
            all_wave.append(samples)
            all_label.append(label)
le = LabelEncoder()
y = le.fit_transform(all_label)      
classes = list(le.classes_) 

y = np_utils.to_categorical(y, num_classes=len(labels))
all_wave = np.array(all_wave).reshape(-1, 8000, 1)

x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                            random_state=777, shuffle=True)
K.clear_session()
inputs = Input(shape=(8000, 1))
conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Flatten()(conv)

conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)
outputs = Dense(len(labels), activation='softmax')(conv)
model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model2.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



import librosa
import IPython.display as ipd
import numpy as np
from keras.models import load_model
import os

model.save('M:/AI_Project/my_model')

from tensorflow import keras
reconstructed_model = keras.models.load_model('M:/AI_Project/my_model')
print(reconstructed_model.summary())

classes=['Benjamin_Netanyau','Jens_Stoltenberg','Julia_Gillard','Magaret_Tarcher','Nelson_Mandela']
def predict(audio):
    prob = reconstructed_model.predict(audio.reshape(1,8000,1))
    index = np.argmax(prob[0])
    return classes[index]

samples1, sample_rate1 = librosa.load('M:/AI_Project/testing/34.wav', sr=16000)
samples1 = librosa.resample(samples1, sample_rate1, 8000)
ipd.Audio(samples1, rate=8000)
print(predict(samples1))