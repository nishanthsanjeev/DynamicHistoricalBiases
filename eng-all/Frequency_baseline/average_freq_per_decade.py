import sqlite3
import json
import random
import numpy as np
from scipy import spatial
from itertools import combinations, cycle
import pandas as pd
from math import sqrt
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import datetime
import sys
pp = PdfPages('avg_frequencies_baseline.pdf')

cnx = sqlite3.connect('/home/nishanthsanjeev/Harvard/DynWE stuff/eng_all_sgns/vectors.decades.db')
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


	filename = sys.argv[1]




	with open(filename, 'r') as f:

		

		for line in f:
			attribute = str(line.rstrip()).split()

			

			plt.ylabel('Average frequency')
			plt.xlabel('Decade')

			plt.title(attribute[0])

			tot = 0.0
			cnt=0
			for word in attribute[1:]:

				try:
					yminval = 1000#used to define the visible ranges in the plot
					ymaxval = -1

					x = []
					y = []


					for table in tablenames:

						
						x.append(table[-4:])

						try:
							freqword = float(freq_list[word][int(table[-4:])])
							cnt+=1
						except KeyError:
							freqword = 0.0

						yminval = min(yminval,freqword)
						ymaxval = max(ymaxval,freqword)
						tot+=freqword
						y.append(freqword)


				except Exception as e:
					print(e)

				if cnt==0:
					avgfreq = 0.0
				else:
					avgfreq = float(tot/cnt)

			result_dict = OrderedDict({
				"Word":word,
				"Category":attribute[0],
				"Average frequency": avgfreq,
				"Decade":table[-4:]
			})

			results.append(result_dict)


			plt.plot(x,y)

			
			plt.yticks(fontsize = 4)
			plt.xticks(fontsize=4)
			plt.ylim(yminval,ymaxval)
			plt.savefig(pp, format = 'pdf')
			plt.clf()	


	pp.close()

df = pd.DataFrame.from_dict(results)
df.to_csv("avg_freq_per_decade_baseline.csv", index = False, sep = '\t')
    
cnx.close()

print(datetime.datetime.now() - begin_time)

# use case:
# $python2 average_freq_per_decade.py python_pachankis.txt

