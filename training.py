import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle # alternative: from sklearn.externals import joblib

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


# -------------------- Preprocessing ----------------------

data = pd.read_csv('F:/merged_data.csv')

def codes_to_binary(iata1, iata2):
    if (str(iata1) == 'nan'):
        fst1 = '11111'
        snd1 = '11111'
        trd1 = '11111'
    else:
        fst1 = '{0:05b}'.format(ord(iata1[0])-65)
        snd1 = '{0:05b}'.format(ord(iata1[1])-65)
        trd1 = '{0:05b}'.format(ord(iata1[2])-65)
    
    if (str(iata2) == 'nan'):
        fst2 = '11111'
        snd2 = '11111'
        trd2 = '11111'
    else:
        fst2 = '{0:05b}'.format(ord(iata2[0])-65)
        snd2 = '{0:05b}'.format(ord(iata2[1])-65)
        trd2 = '{0:05b}'.format(ord(iata2[2])-65)
    
    return [int(x) for x in list(fst1+snd1+trd1+fst2+snd2+trd2)]
    

binary_row = []
for index, row in data.iterrows():
    schd_from = row['schd_from']
    schd_to = row['schd_to']
    binary_row.append(codes_to_binary(schd_from, schd_to))
del index, row

scaler = StandardScaler()
X = scaler.fit_transform(data.iloc[:, 0:5])
X = np.concatenate((X, binary_row), axis=1)
del data, binary_row


X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)


# -------------------- AI ----------------------

input_dim = X_train.shape[1]
encoding_dim = 25

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


nb_epoch = 100
batch_size = 32

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

#checkpointer = ModelCheckpoint(filepath="model.h5",
#                               verbose=0,
#                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    #validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[tensorboard]).history


# Run the following in the terminal to start TensorBoard:
# tensorboard --logdir=C:\Users\Simon\Downloads\secure.flightradar24.com\scripts\logs