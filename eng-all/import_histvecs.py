import sqlite3
import json

cnx = sqlite3.connect('vectors.decades.db')
cnx.text_factory = str

cur = cnx.cursor()
cur.execute('DROP TABLE IF EXISTS vectors1800;')
cur.execute('DROP TABLE IF EXISTS vectors1810;')
cur.execute('DROP TABLE IF EXISTS vectors1820;')
cur.execute('DROP TABLE IF EXISTS vectors1830;')
cur.execute('DROP TABLE IF EXISTS vectors1840;')
cur.execute('DROP TABLE IF EXISTS vectors1850;')
cur.execute('DROP TABLE IF EXISTS vectors1860;')
cur.execute('DROP TABLE IF EXISTS vectors1870;')
cur.execute('DROP TABLE IF EXISTS vectors1880;')
cur.execute('DROP TABLE IF EXISTS vectors1890;')
cur.execute('DROP TABLE IF EXISTS vectors1900;')
cur.execute('DROP TABLE IF EXISTS vectors1910;')
cur.execute('DROP TABLE IF EXISTS vectors1920;')
cur.execute('DROP TABLE IF EXISTS vectors1930;')
cur.execute('DROP TABLE IF EXISTS vectors1940;')
cur.execute('DROP TABLE IF EXISTS vectors1950;')
cur.execute('DROP TABLE IF EXISTS vectors1960;')
cur.execute('DROP TABLE IF EXISTS vectors1970;')
cur.execute('DROP TABLE IF EXISTS vectors1980;')
cur.execute('DROP TABLE IF EXISTS vectors1990;')


cur.execute('CREATE TABLE vectors1800 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1810 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1820 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1830 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1840 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1850 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1860 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1870 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1880 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1890 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1900 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1910 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1920 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1930 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1940 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1950 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1960 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1970 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1980 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')
cur.execute('CREATE TABLE vectors1990 (word VARCHAR(255), vector TEXT, PRIMARY KEY (word));')


tablenames = ["vectors1800", "vectors1810", "vectors1820", "vectors1830", "vectors1840", "vectors1850", "vectors1860", "vectors1870", "vectors1880", "vectors1890", "vectors1900", "vectors1910", "vectors1920", "vectors1930", "vectors1940", "vectors1950", "vectors1960", "vectors1970", "vectors1980", "vectors1990"]
filenames = ["vectors1800.txt", "vectors1810.txt", "vectors1820.txt", "vectors1830.txt", "vectors1840.txt", "vectors1850.txt", "vectors1860.txt", "vectors1870.txt", "vectors1880.txt", "vectors1890.txt", "vectors1900.txt", "vectors1910.txt", "vectors1920.txt", "vectors1930.txt", "vectors1940.txt", "vectors1950.txt", "vectors1960.txt", "vectors1970.txt", "vectors1980.txt", "vectors1990.txt"]



for tablename, filename in zip(tablenames, filenames):
    with open(filename, 'r',encoding = 'utf-8', errors='ignore') as f:
        jobs = 0
        for line in f:
            row = line.rstrip().split(' ')
            word = str(row[0])
            if len(word) <= 255:
                vector = [float(val) for val in row[1:]]
                s_vector = json.dumps(vector)
                sql = 'INSERT INTO {} (word, vector) VALUES (?, ?)'.format(
                    tablename)
                insert_data = (word, s_vector)
                cur.execute(sql, insert_data)
                jobs += 1
                if jobs % 1000000 == 0:
                    print('Jobs done: {0}'.format(jobs))
            else:
                print('one exclusion: ', word)
    cnx.commit()

    cur.execute('SELECT count(word), count(vector) FROM {};'.format(tablename))
    print(cur.fetchall())

cnx.close()

## cd /Users/tessacharlesworth/Dropbox/1\ -\ Research\ Projects/22\ -\ Hist\ Embeddings/vectors/normalized_clean/sgns 
## python3 import_histvecs.py
