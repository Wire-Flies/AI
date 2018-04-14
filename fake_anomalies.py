import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from generator1 import train_generator1, val_generator1
from avgs_and_stds import avgs, stds

def swap_schd(gen):
    g = next(gen)[0]
    g_original = g.copy()
    
    from_lat, from_long = g[0][5:7]
    to_lat, to_long = g[0][7:]
    g[0][5] = to_lat
    g[0][6] = to_long
    g[0][7] = from_lat
    g[0][8] = from_long
    return (g_original, g)

def change_direction(gen):
    g = next(gen)[0]
    g_original = g.copy()
    
    direction = g[0][3] * stds[3] + avgs[3]
    direction = (direction + 90) % 360 # 90 degree rotation
    direction = (direction - avgs[3]) / stds[3]
    g[0][3] = direction
    return (g_original, g)

def random_pos(gen):
    g = next(gen)[0]
    g_original = g.copy()
    
    lat, long = g[0][0:2]
    g[0][0] = (50 + np.random.randint(18) - avgs[2]) / stds[2]
    g[0][1] = (np.random.randint(35) - avgs[2]) / stds[2]
    return (g_original, g)

def halve_altitude(gen):
    g = next(gen)[0]
    g_original = g.copy()
    
    alt = g[0][2] * stds[2] + avgs[2]
    alt = np.random.randint(avgs[2] * 2)
    g[0][2] = (alt - avgs[2]) / stds[2]
    return (g_original, g)


gen = train_generator1(1)
autoencoder = load_model('./models/model4.h5')

diffs1 = []
diffs2 = []
diffs3 = []
diffs4 = []
for i in range(10000):
    g_original, g = swap_schd(gen)
    original_error = mean_squared_error(g_original, autoencoder.predict(g_original))
    modified_error = mean_squared_error(g, autoencoder.predict(g))
    diff = modified_error / original_error
    diffs1.append(diff)
    
    g_original, g = change_direction(gen)
    original_error = mean_squared_error(g_original, autoencoder.predict(g_original))
    modified_error = mean_squared_error(g, autoencoder.predict(g))
    diff = modified_error / original_error
    diffs2.append(diff)
    
    g_original, g = random_pos(gen)
    original_error = mean_squared_error(g_original, autoencoder.predict(g_original))
    modified_error = mean_squared_error(g, autoencoder.predict(g))
    diff = modified_error / original_error
    diffs3.append(diff)
    
    g_original, g = halve_altitude(gen)
    original_error = mean_squared_error(g_original, autoencoder.predict(g_original))
    modified_error = mean_squared_error(g, autoencoder.predict(g))
    diff = modified_error / original_error
    diffs4.append(diff)
print("swap_schd")
print(np.mean(diffs1))
print(np.std(diffs1))
print()
print("change_direction")
print(np.mean(diffs2))
print(np.std(diffs2))
print()
print("random_pos")
print(np.mean(diffs3))
print(np.std(diffs3))
print()
print("halve_altitude")
print(np.mean(diffs4))
print(np.std(diffs4))
print()
