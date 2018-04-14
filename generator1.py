import numpy as np

from generator2 import train_generator2, val_generator2

# trains on 9/10ths of the data
def train_generator1(batch_size):
    array_size = 100000
    gen2 = train_generator2(array_size)
    while True:
        from_database = next(gen2)
        perm = np.random.permutation(array_size)
        batch = np.empty((batch_size, 9), dtype=float)
        counter = 0
        for p in perm:
            batch[counter] = from_database[p]
            if counter == batch_size - 1:
                yield (batch, batch)
                counter = 0
            else:
                counter += 1
        

# validates on 1/10th of the data
def val_generator1(batch_size):
    array_size = 100000
    gen2 = val_generator2(array_size)
    while True:
        from_database = next(gen2)
        perm = np.random.permutation(array_size)
        batch = np.empty((batch_size, 9), dtype=float)
        counter = 0
        for p in perm:
            batch[counter] = from_database[p]
            if counter == batch_size - 1:
                yield (batch, batch)
                counter = 0
            else:
                counter += 1