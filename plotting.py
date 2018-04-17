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
from avgs_and_stds import avgs, stds

def unscale(a):
    b = []
    for i, x in enumerate(a):
        b.append(x * stds[i] + avgs[i])
    return b

autoencoder = load_model('models/model6.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_2').output)

gen = train_generator1(1000000)
data = next(gen)[0]
encoded_data = encoder.predict(data)
np.savetxt('encoded_data.tsv', encoded_data, fmt='%.5f', delimiter='\t')
np.savetxt('meta_data.tsv', list(map(unscale, data)), fmt='%.5f', delimiter='\t', header='lat\tlong\talt\theading\tspeed\tlat_from\tlong_from\tlat_to\tlong_to')
# Upload file to https://projector.tensorflow.org/



# --------- Plots ---------

mu = encoded_data.mean(axis=0)
U,s,V = np.linalg.svd(encoded_data - mu, full_matrices=False)
Zpca = np.dot(encoded_data - mu, V.transpose())

Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
err = np.sum((encoded_data-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

Zenc = encoder.predict(data)  # bottleneck representation
Renc = autoencoder.predict(data)        # reconstruction

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title('PCA')
plt.scatter(Zpca[:5000,0], Zpca[:5000,1], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.subplot(122)
plt.title('Autoencoder')
plt.scatter(Zenc[:5000,0], Zenc[:5000,1], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.tight_layout()

# Source: https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca/292516