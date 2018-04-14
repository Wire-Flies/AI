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

input_layer = Input(shape=(9, ))
encoder = Dense(6, activation="tanh", activity_regularizer=regularizers.l1(10e-2))(input_layer)
middle = Dense(3, activation="tanh")(encoder)
decoder = Dense(6, activation="tanh")(middle)
decoder = Dense(9)(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
#encoder = Model(inputs=input_layer, outputs=middle)


nb_epoch = 100
batch_size = 100

optimizer = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=optimizer, 
                    loss='mean_squared_error', 
                    #metrics=['accuracy']
                    )


# model2: 9 8 7 6 4 6 7 8 9, max: 1.02, min: 0.00062
# model3: 9 8 7 6 3 6 7 8 9, max: 1.4, min: ?, avg: 0.07
# model4: 9 8 7 6 - 6 7 8 9, max: , min: , avg: 
# model5: 9 8 7 6 5 6 7 8 9, max: 1.02, min: 0.00078, avg: 0.048
# model6: 9, 7 (AR), 5, 7, 9 (accidentally overwrote model 5)
checkpointer = ModelCheckpoint(filepath="./models/model7.h5",
                               verbose=1,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit_generator(train_generator1(batch_size),
                    epochs=nb_epoch,
                    steps_per_epoch=7500,
                    validation_data=val_generator1(batch_size),
                    validation_steps=5000,
                    verbose=1,
                    #initial_epoch=0,
                    #max_queue_size=1,
                    #workers=3, 
                    #use_multiprocessing=False,
                    callbacks=[tensorboard, checkpointer]).history


# Run the following in the terminal to start TensorBoard:
# tensorboard --logdir=path-to-logs-directory