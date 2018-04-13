from keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np
import psycopg2
from avgs_and_stds import avgs, stds
from generator1 import train_generator1, val_generator1


autoencoder = load_model('model.h5')

in_vec = [55.0508738124329, # test
        12.477793190734,
        12534.366649766861,
        175.1113875125417309,
        242.8279644653325668,
        45.0933818074002,
        12.5955883176537,
        44.7382849382711,
        11.2354800043253]

out_vec = autoencoder.predict(in_vec)




autoencoder = load_model('model.h5')

conn = psycopg2.connect("dbname=wireflies user=wireflies password=wireflies")
cur = conn.cursor()

gen = val_generator1(100000)
g = next(gen)[0]
result = []
for h in g:
    #test_in_unscaled = list(h); print(h)
    #test_in_scaled = np.array([scale(test_in_unscaled)])
    h = np.array([h])
    test_out_scaled = autoencoder.predict(h)
    test_out_unscaled = unscale(list(test_out_scaled[0]))
    
    result.append(mean_squared_error(h, test_out_scaled))
    
#    print("in:", test_in_unscaled)
#    print("out:", test_out_unscaled)
#    print("norm:", mean_squared_error(test_in_scaled, test_out_scaled))
#    print()
print(max(result))
print(min(result))
print(np.mean(result))
    
    
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