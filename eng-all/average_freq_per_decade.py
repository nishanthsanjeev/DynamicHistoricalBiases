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
import datetime
pp = PdfPages('average_frequencies.pdf')

cnx = sqlite3.connect('vectors.decades.db')
cnx.text_factory = str
np.random.seed(111)

cur = cnx.cursor()

memorybank = {}


if __name__ == '__main__':

	begin_time = datetime.datetime.now()


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

		x = []
		y = []
		yminval = 1000
		ymaxval = -1
		for line in f:
			attribute = str(line.rstrip()).split()

			try:

				minval = 1000
				maxval = -1

				

				tot = 0

				for table in tablenames:

					cnt = 0
					val = 0


					for word in attribute:
						try:
							addval = float(freq_list[word][int(table[-4:])])
						except KeyError:
							addval = 0.0

						minval = min(minval,addval)
						maxval = max(maxval,addval)

						if addval==0.0:
							#print(word+" "+table[-4:])
							continue
								
						
						val+= addval
						cnt+=1

					if cnt==0:
						avg = 0.0
					else:
						avgw = (float)(val/cnt)
					
					tot+=avgw
					

				tot/=20


			except Exception as e:
				print(e)

			y.append(attribute[0])
			x.append(tot)
			yminval = min(yminval,tot)
			ymaxval = max(ymaxval,tot)



			result_dict = OrderedDict({
					"Label":attribute[0],
					"Relative_frequency": tot,
					"Max_val": maxval,
					"Min_val": minval
				})

			results.append(result_dict)


	plt.plot(x,y)
	plt.ylabel('Test')
	plt.xlabel('Relative frequency')

	plt.title("Labels")
	plt.yticks(fontsize = 4)
	
	plt.xlim(yminval,ymaxval)
	
	plt.savefig(pp, format = 'pdf')
	plt.clf()				

	pp.close()

df = pd.DataFrame.from_dict(results)
df.to_csv("avg_freq_per_decade.csv", index = True)
    
cnx.close()

print(datetime.datetime.now() - begin_time)

