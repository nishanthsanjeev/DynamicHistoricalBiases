import sqlite3
import json
import random
import numpy as np
from scipy import spatial
from itertools import combinations
import pandas as pd
from math import sqrt
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
pp = PdfPages('frequencies.pdf')

cnx = sqlite3.connect('vectors.decades.db')
cnx.text_factory = str
np.random.seed(111)

cur = cnx.cursor()

memorybank = {}


if __name__ == '__main__':

	vecs = {
		
		"good": "love cheer friend pleasure adore cheerful friendship joyful smiling cherish excellent glad joyous spectacular appealing delight excitement laughing attractive delightful fabulous glorious pleasing beautiful fantastic happy lovely terrific celebrate enjoy magnificent triumph paradise splendid affectionate euphoric good commendable positive amazing better best favorable".split(),
		"bad": "abuse grief poison sadness pain despise failure nasty angry detest horrible negative ugly dirty gross evil rotten annoy disaster horrific scorn awful disgust hate humiliate selfish tragic bothersome hatred hurtful sickening yucky ghastly repulsive aggravate horrendous bad ill worst vicious immoral violation violence repugnant pernicious unthankful detriment deleterious unpleasant unfavorable terrible".split()
	}


	results = []

	
	tablenames = ["vectors1800", "vectors1810", "vectors1820", "vectors1830", "vectors1840", "vectors1850",
	 "vectors1860", "vectors1870", "vectors1880", "vectors1890", "vectors1900", "vectors1910", 
	 "vectors1920", "vectors1930", "vectors1940", "vectors1950", "vectors1960", "vectors1970", 
	 "vectors1980", "vectors1990"]
	

	with open('/home/nishanthsanjeev/Harvard/Frequency script/freqs.pkl','r') as fre:
		freq_list = pickle.load(fre)



	with open("python_pachankis.txt", 'r') as f:
		f.readline()
		for line in f:
			attribute = str(line.rstrip()).split()

			try:

				x = []
				y = []

				yminval = 1000
				ymaxval = -1

				for table in tablenames:
					
					x.append(table[-4:])
					
					y.append(float(freq_list[attribute[0]][int(table[-4:])]))
					
					print("{:.11f}".format(freq_list[attribute[0]][int(table[-4:])]))
					print(attribute[0])
					print(int(table[-4:]))


					yminval = min(yminval,float(freq_list[attribute[0]][int(table[-4:])]))
					ymaxval = max(ymaxval,float(freq_list[attribute[0]][int(table[-4:])]))


				plt.plot(x,y)
				plt.xlabel('Year')
				plt.ylabel('Relative frequency')

				plt.title(attribute[0])
				plt.xticks(rotation = 90)

				plt.ylim(yminval,ymaxval)



				plt.savefig(pp, format = 'pdf')
				plt.clf()
	

			except Exception as e:
				print(e)				

	pp.close()

    
cnx.close()
