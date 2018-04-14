import psycopg2
import numpy as np
from avgs_and_stds import avgs, stds

# trains on 9/10ths of the data
def train_generator2(array_size):
    conn = psycopg2.connect("dbname=wireflies user=wireflies password=wireflies")
    cur = conn.cursor()
    cur.execute("SELECT max(id) FROM flight_data_new")
    max_id = cur.fetchone()[0]
    while True:
        perm = np.random.permutation(int(max_id / array_size) - 1)
        for p in perm:
            if p % 10 == 0:
                continue
            batch = np.empty((array_size, 9), dtype=float)
            r = p * array_size
            cur.execute(f"SELECT fd.latitude, fd.longitude, fd.altitude, fd.heading, fd.speed, \
                        cast(a.latitude_deg AS real), cast(a.longitude_deg AS real), \
                        cast(b.latitude_deg AS real), cast(b.longitude_deg AS real)\
                        	FROM flight_data_new fd\
                        	NATURAL JOIN flights f\
                        	LEFT JOIN airports a ON a.iata_code = f.schd_from\
                        	LEFT JOIN airports b ON b.iata_code = f.schd_to\
                        	WHERE fd.id BETWEEN {r} AND {r + array_size}")
            row = cur.fetchmany(array_size)
            for i, r in enumerate(row):
                r = [(0 - avgs[j]) / stds[j] if x == None else 
                     (x - avgs[j]) / stds[j] 
                     for j, x in enumerate(r)]
                #x = (x - avgs[j]) / stds[j] # scaling
                batch[i] = r
            yield batch
    

# validates on 1/10th of the data
def val_generator2(array_size):
    conn = psycopg2.connect("dbname=wireflies user=wireflies password=wireflies")
    cur = conn.cursor()
    cur.execute("SELECT max(id) FROM flight_data_new")
    max_id = cur.fetchone()[0]
    while True:
        perm = np.random.permutation(int(max_id / array_size) - 1)
        for p in perm:
            if p % 10 != 0:
                continue
            batch = np.empty((array_size, 9), dtype=float)
            r = p * array_size
            cur.execute(f"SELECT fd.latitude, fd.longitude, fd.altitude, fd.heading, fd.speed, \
                        cast(a.latitude_deg AS real), cast(a.longitude_deg AS real), \
                        cast(b.latitude_deg AS real), cast(b.longitude_deg AS real)\
                        	FROM flight_data_new fd\
                        	NATURAL JOIN flights f\
                        	LEFT JOIN airports a ON a.iata_code = f.schd_from\
                        	LEFT JOIN airports b ON b.iata_code = f.schd_to\
                        	WHERE fd.id BETWEEN {r} AND {r + array_size}")
            row = cur.fetchmany(array_size)
            for i, r in enumerate(row):
                r = [(0 - avgs[j]) / stds[j] if x == None else 
                     (x - avgs[j]) / stds[j] 
                     for j, x in enumerate(r)]
                #x = (x - avgs[j]) / stds[j] # scaling
                batch[i] = r
            yield batch
        
