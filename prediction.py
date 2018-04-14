from keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np
import psycopg2
import seaborn as sns
from avgs_and_stds import avgs, stds
from generator1 import train_generator1, val_generator1


def scale(a):
    b = []
    for i, x in enumerate(a):
        b.append((x - avgs[i]) / stds[i])
    return b
    
def unscale(a):
    b = []
    for i, x in enumerate(a):
        b.append(x * stds[i] + avgs[i])
    return b


autoencoder = load_model('./models/model6.h5')

conn = psycopg2.connect("dbname=wireflies user=wireflies password=wireflies")
cur = conn.cursor()

gen = val_generator1(10000)
g = next(gen)[0]
result = []
for h in g:
    #test_in_unscaled = list(h); print(h)
    #test_in_scaled = np.array([scale(test_in_unscaled)])
    h = np.array([h])
    test_out_scaled = autoencoder.predict(h)
    test_out_unscaled = unscale(list(test_out_scaled[0]))
    result.append(mean_squared_error(h, test_out_scaled))
    
print(max(result))
print(min(result))
print(np.mean(result))
sns.distplot(result)
    
