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
pp = PdfPages('engfic_ind_frequencies.pdf')

cnx = sqlite3.connect('/home/nishanthsanjeev/Harvard/DynWE stuff/WEAT EngFic/vectors.decades.db')
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

	#tablenames = ["vectors1800", "vectors1810", "vectors1820"]
	

	with open('/media/nishanthsanjeev/2CC66AE5C66AAF30/Embeddings/eng-fiction-all/freqs.pkl','r') as fre:
		freq_list = pickle.load(fre)


	filename = sys.argv[1]

	# cmap = plt.get_cmap('gnuplot')
	# colors = [cmap(i) for i in np.linspace(0, 200, 5)]
	number_of_colors = 25
	cycol = cycle(['black','dimgray','red','peru','saddlebrown','darkorange','gold','olive','yellowgreen','lawngreen','darkseagreen','forestgreen','aquamarine','deepskyblue','mediumblue','slateblue','darkviolet','violet','deeppink','pink','lightcoral','teal'])

	#color = ["#"+''.join([random.sample('0123456789ABCDEF',6)]) for i in range(number_of_colors)]


	with open(filename, 'r') as f:

		

		for line in f:
			attribute = str(line.rstrip()).split()

			yminval = 1000#used to define the visible ranges in the plot
			ymaxval = -1

			plt.ylabel('Frequency')
			plt.xlabel('Decade')

			plt.title(attribute[0])

			colcnt = 1

			for word in attribute[1:]:
				try:

					x = []
					y = []


					for table in tablenames:

						#x.append(int(table[-4:]))
						x.append(table[-4:])

						try:
							freqword = float(freq_list[word][int(table[-4:])])
						except KeyError:
							freqword = 0.0

						yminval = min(yminval,freqword)
						ymaxval = max(ymaxval,freqword)
						y.append(freqword)

						#print(int(table[-4:]))

						result_dict = OrderedDict({
								"Word":word,
								"Category":attribute[0],
								"Frequency": freqword,
								"Decade":table[-4:]
							})

						results.append(result_dict)


					#plt.plot(x,y, color = color[colcnt], label = word)
					plt.plot(x,y, color = cycol.next(), label = word)
					#plt.plot(x,y,label = word)
					colcnt+=1
								


				except Exception as e:
					print(e)


			# result_dict = OrderedDict({
			# 		"Label":attribute[0],
			# 		"Relative_frequency": tot,
			# 		"Max_val": maxval,
			# 		"Min_val": minval
			# 	})

			# results.append(result_dict)

			
			plt.yticks(fontsize = 4)
			plt.xticks(fontsize=4)
			plt.ylim(yminval,ymaxval)
			plt.legend(loc='upper left', fontsize='xx-small')
			plt.savefig(pp, format = 'pdf')
			plt.clf()	


	pp.close()

df = pd.DataFrame.from_dict(results)
df.to_csv("engfic_ind_freq_per_decade.csv", index = False, sep = '\t')
    
cnx.close()

print(datetime.datetime.now() - begin_time)

# use case:
# $python2 average_freq_per_decade.py python_pachankis.txt