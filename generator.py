import psycopg2
import random
import math
import numpy as np

# trains on 9/10ths of the data
def train_generator(batch_size):
    conn = psycopg2.connect("dbname=wireflies user=wireflies password=wireflies")
    cur = conn.cursor()
    cur.execute("SELECT max(id) FROM flight_data_new")
    max_id = cur.fetchone()[0]
    while True:
        batch = []
        for i in range(batch_size):
            r = 0
            while r % 10 == 0:
                r = random.randint(1, max_id)
            cur.execute(f"SELECT fd.latitude, fd.longitude, fd.altitude, fd.heading, fd.speed, \
                        cast(a.latitude_deg AS real), cast(a.longitude_deg AS real), \
                        cast(b.latitude_deg AS real), cast(b.longitude_deg AS real)\
                    	FROM flight_data_new fd\
                    	NATURAL JOIN flights f\
                    	LEFT JOIN airports a ON a.iata_code = f.schd_from\
                    	LEFT JOIN airports b ON b.iata_code = f.schd_to\
                    	WHERE fd.id = {r}")
            row = cur.fetchone()
            row = [0 if x == None else x for x in row]
            batch.append(row)
        yield (np.array(batch), np.array(batch))
        

# validates on 1/10th of the data
def val_generator(batch_size):
    conn = psycopg2.connect("dbname=wireflies user=wireflies password=wireflies")
    cur = conn.cursor()
    cur.execute("SELECT max(id) FROM flight_data_new")
    max_id = cur.fetchone()
    while True:
        batch = []
        for i in range(batch_size):
            r = random.randint(1, max_id)
            r = min(int(math.ceil(r / 10.0)) * 10, max_id)
            cur.execute(f"SELECT fd.latitude, fd.longitude, fd.altitude, fd.heading, fd.speed, \
                        cast(a.latitude_deg AS real), cast(a.longitude_deg AS real), \
                        cast(b.latitude_deg AS real), cast(b.longitude_deg AS real)\
                    	FROM flight_data_new fd\
                    	NATURAL JOIN flights f\
                    	LEFT JOIN airports a ON a.iata_code = f.schd_from\
                    	LEFT JOIN airports b ON b.iata_code = f.schd_to\
                    	WHERE fd.id = {r}")
            row = cur.fetchone()
            row = [0 if x == None else x for x in row]
            batch.append(row)
        yield (np.array(batch), np.array(batch))
        