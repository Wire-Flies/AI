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
from keras import optimizers

from generator1 import train_generator1, val_generator1


# -------------------- AI ----------------------

#input_dim = 9
#encoding_dim = 7

input_layer = Input(shape=(9, ))
encoder = Dense(7, activation="tanh", kernel_initializer='uniform')(input_layer) 
                #activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(5, activation="tanh", kernel_initializer='uniform')(encoder)
decoder = Dense(7, activation="tanh", kernel_initializer='uniform')(encoder)
decoder = Dense(9, kernel_initializer='uniform')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


nb_epoch = 100
batch_size = 100

optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=optimizer, 
                    loss='mean_squared_error', 
                    #metrics=['accuracy']
                    )

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=1,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit_generator(train_generator1(batch_size),
                    epochs=nb_epoch,
                    steps_per_epoch=10000,
                    validation_data=val_generator1(batch_size),
                    validation_steps=1000,
                    verbose=1,
                    #max_queue_size=1,
                    #workers=3, 
                    #use_multiprocessing=False,
                    callbacks=[tensorboard, checkpointer]).history


# Run the following in the terminal to start TensorBoard:
# tensorboard --logdir=path-to-logs-directory