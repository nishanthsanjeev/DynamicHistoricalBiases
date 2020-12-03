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
import sys
pp = PdfPages('Avg_EngAll_WEAT_Baseline.pdf')

cnx = sqlite3.connect('/home/nishanthsanjeev/Harvard/DynWE stuff/eng_all_sgns/vectors.decades.db')
cnx.text_factory = str
np.random.seed(111)

cur = cnx.cursor()

memorybank = {}

def weighted_std(values, weights):
	"""
	Return the weighted standard deviation.

	values, weights -- Numpy ndarrays with the same shape.
	"""
	average = np.average(values, weights=weights)
	# Fast and numerically precise:
	variance = np.average((values-average)**2, weights=weights)
	# Small sample size bias correction:
	variance_ddof1 = variance*len(values)/(len(values)-1)

	return float(sqrt(variance_ddof1))

def within_group_cohesion(X):
	dist = spatial.distance.pdist(X, 'cosine')
	return dist.mean()

def group_cohesion_test(X, Y, perm_n = 1000, permtype = 1):


	test_statistic = np.average((within_group_cohesion(X), within_group_cohesion(Y)),
		weights = (len(X), len(Y)))
	jointlist = np.concatenate((X,Y))
	permutations = np.array([])
	if permtype == 1:
		count = 0
		cutpoint = len(X)
		while count < perm_n:
			np.random.shuffle(jointlist)
			set1 = jointlist[:cutpoint]
			set2 = jointlist[cutpoint:]

			permutations = np.append(permutations, 
				np.average([within_group_cohesion(set1), within_group_cohesion(set2)], 
					weights = [len(set1), len(set2)])
			)
			count += 1
	else:
		nums = list(range(len(jointlist)))
		for comb in combinations(nums, len(X)):
			set1 = [item for i, item in enumerate(jointlist) if i in comb]
			set2 = [item for i, item in enumerate(jointlist) if i not in comb]

			permutations = np.append(permutations, 
				np.average([within_group_cohesion(set1), within_group_cohesion(set2)], 
					weights = [len(set1), len(set2)])
			)
	P_val = (sum(i <= test_statistic for i in permutations)+1)/(len(permutations)+1)
	return P_val



def s_weat(X, A, B, permt = 0, perm_n = 10000, table = "vectors1800", cohesion_test = False, 
cohesion_permutations = 1000, cohesion_type = 2):

	def diff_sim(X, A, B):

		sum_A = 0
		sum_B = 0

		if len(X)==0 or len(A)==0 or len(B)==0:
			return 0

		all_sims = []
		for a in A:
			a_ = a.reshape(1, -1)
			results = spatial.distance.cdist(a_, X, 'cosine')
			results_filtered = np.array([result for result in results[0] if (result == result)])
			sum_X = (1 - results_filtered).sum()
			val = sum_X/len(X)
			sum_A += val
			all_sims.append(val)
		ave_A = sum_A/len(A)

		for b in B:
			b_ = b.reshape(1, -1)
			results = spatial.distance.cdist(b_, X, 'cosine')
			results_filtered = np.array([result for result in results[0] if (result == result)])
			sum_X = (1 - results_filtered).sum()
			val = sum_X/len(X)
			sum_B += val
			all_sims.append(val)
		ave_B = sum_B/len(B)

		difference = ave_A - ave_B

		# For SD calculation, assign weights based on frequency of opposite category
		weights = [len(B) for num in range(len(A))] + [len(A) for num in range(len(B))]
		standard_dev = weighted_std(all_sims, weights)


		if difference==0 or standard_dev==0:
			return 0
		effect_size = difference/standard_dev

		
		return effect_size


	def permutation_test(X, A, B):
		jointlist = np.array(list(A) + list(B))
		permutations = []
		nums = list(range(len(jointlist)))
		for comb in combinations(nums, len(A)):
			set1 = [item for i, item in enumerate(jointlist) if i in comb]
			set2 = [item for i, item in enumerate(jointlist) if i not in comb]
			permutations.append(diff_sim(X, set1, set2))
		return permutations

	def rand_test(X, A, B, perm_n):
		jointlist = np.array(list(A) + list(B))
		
		np.random.shuffle(jointlist)
		permutations = []
		count = 0
		cutpoint = len(A)
		while count < perm_n:
			np.random.shuffle(jointlist)
			set1 = jointlist[:cutpoint]
			set2 = jointlist[cutpoint:]
			permutations.append(diff_sim(X, set1, set2))
			count += 1
		return permutations

	allwords = list(X + A + B)
	
	if table not in memorybank.keys():
		memorybank[table] = {}
	cache = [i for i in allwords if i in memorybank[table].keys()]
	nocache = [i for i in allwords if i not in memorybank[table].keys()]

	if len(nocache) > 0:
		format_strings = ','.join(['?'] * len(nocache))
		cur.execute('SELECT word, vector FROM {0} WHERE word IN ({1})'.format(table, format_strings), tuple(nocache))
		Results = [(x[0], json.loads(x[1])) for x in cur.fetchall()]
		for item in Results:
			memorybank[table][item[0]] = (item[0], item[1])
	else:
		Results = []
	if len(cache) > 0:
		for item in cache:
			Results.append(memorybank[table][item])
	Cat1 = np.array([word[1] for word in Results if word[0] in X])
	Att1 = np.array([word[1] for word in Results if word[0] in A])
	Att2 = np.array([word[1] for word in Results if word[0] in B])

	effect_size = diff_sim(Cat1, Att1, Att2)

	missing = [word for word in allwords if word not in [res[0] for res in Results]]
	result_dict = OrderedDict({
		"table": table,
		# the category labels are not fed to the function, so assign them at higher level upon return
		"categories": None,
		"attribute(s)": None,
		"effect_size": effect_size,
		"missing": missing
	})

	return result_dict


if __name__ == '__main__':

	begin_time = datetime.datetime.now()

	vecs = {

		"good": "love cheer friend pleasure adore cheerful friendship joyful smiling cherish excellent glad joyous spectacular appealing delight excitement laughing attractive delightful fabulous glorious pleasing beautiful fantastic happy lovely terrific celebrate enjoy magnificent triumph paradise splendid affectionate euphoric good commendable positive amazing better best favorable".split(),
		"bad": "abuse grief poison sadness pain despise failure nasty angry detest horrible negative ugly dirty gross evil rotten annoy disaster horrific scorn awful disgust hate humiliate selfish tragic bothersome hatred hurtful sickening yucky ghastly repulsive aggravate horrendous bad ill worst vicious immoral violation violence repugnant pernicious unthankful detriment deleterious unpleasant unfavorable terrible".split()
	}


	results = []

	filename = sys.argv[1]

	tablenames = ["vectors1800", "vectors1810", "vectors1820", "vectors1830", "vectors1840", "vectors1850",
	 "vectors1860", "vectors1870", "vectors1880", "vectors1890", "vectors1900", "vectors1910", 
	 "vectors1920", "vectors1930", "vectors1940", "vectors1950", "vectors1960", "vectors1970", 
	 "vectors1980", "vectors1990"]

	with open(filename, 'r') as f:
		#f.readline()

		x = []
		y = []

		for line in f:
			attribute = str(line.rstrip()).split()

			maxval = -5
			minval = 1000

			avgval = 0
			try:

				for table in tablenames:
					
					result_dict = s_weat(attribute[1:], vecs["good"], vecs["bad"],  # changed attribute to a "vector" here
						permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
					result_dict["categories"] = "good vs. bad"
					result_dict["attribute(s)"] = attribute[0] # here, we will want just the first item from each row of the attribute text document
					avgval+=result_dict["effect_size"]
					minval = min(minval,result_dict["effect_size"])
					maxval = max(maxval,result_dict["effect_size"])

			except Exception as e:
				print(e)


			avgval/=20
			y.append(attribute[0])
			x.append(avgval)

			result_dict2 = OrderedDict({
					"Label": attribute[0],
					"Valence": avgval,
					"Min_val": minval,
					"Max_val": maxval
				})

			results.append(result_dict2)



	plt.plot(x,y)
	plt.ylabel('Test')
	plt.xlabel('Effect_size')
	plt.title("Valence")
	plt.yticks(fontsize = 4)
	plt.savefig(pp, format = 'pdf')
	plt.clf()
					
				

	pp.close()

	df = pd.DataFrame.from_dict(results)
	df.to_csv("avg_weat_baseline.csv", index = False, sep='\t')
    
    
cnx.close()


print(datetime.datetime.now() - begin_time)

#use case
#$python avg_weat_all_decades.py python_pachankis.txt
