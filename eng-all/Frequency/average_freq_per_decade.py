'''

This script is used to find the "average frequency" of each category. The process is as follows:

For each category, iterate through its stimuli list, and find the individual frequencies of each word, per decade.


Let's say, a particular category has 20 words. For each of these words, retrieve their average frequency across all decades.

Finally, sum those frequencys to get the average frequency of the category occurring in the corpus.


'''

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
pp = PdfPages('avg_frequencies.pdf')
np.random.seed(111)


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
	

	with open('/../../freqs.pkl','r') as fre:
		freq_list = pickle.load(fre)#Stores historical word frequencies per decade; retrieved from HistWords



	with open('/../python_pachankis.txt', 'r') as f:	

		for line in f:
			attribute = str(line.rstrip()).split()

			

			plt.ylabel('Average frequency')
			plt.xlabel('Decade')

			plt.title(attribute[0])

			tot = 0.0
			cnt=0
			catfreq=0.0
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
							cnt+=1#keeps track of number of decades where 'word' is found in the corpus
						except KeyError:#if 'word' is not found for this particular decade
							freqword = 0.0

						yminval = min(yminval,freqword)
						ymaxval = max(ymaxval,freqword)
						tot+=freqword#Adds on the individual frequency contributed by each word
						y.append(freqword)

					if cnt==0:
						avgfreq = 0.0
					else:
						avgfreq = float(tot/cnt)#the average frequency of 'word' across all decades

					catfreq+=avgfreq


				except Exception as e:
					print(e)

				

			result_dict = OrderedDict({
				"Category":attribute[0],
				"Average frequency": catfreq,
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
df.to_csv("avg_freq_per_decade.csv", index = False, sep = '\t')

print(datetime.datetime.now() - begin_time)

# use case:
# $python2 average_freq_per_decade.py
