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


cnx = sqlite3.connect('vectors.decades.db')
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
	return sqrt(variance_ddof1)

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

		effect_list = []

		sum_A = 0
		sum_B = 0

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

	### The following code snippet is used to create a cache of word vectors for fast access ###

	### BEGINS HERE ###

	if table not in memorybank.keys():#if the word vector table for a particular year has not been encountered yet, initialize a new space for it in memorybank
		memorybank[table] = {}
	cache = [i for i in allwords if i in memorybank[table].keys()]#if a word is present in memorybank, add to cache
	nocache = [i for i in allwords if i not in memorybank[table].keys()]#if a word is not present in memorybank, add to nocache

	if len(nocache) > 0:#Words that are in nocache have not been encountered yet, and so we must retrieve them from the database and store in cache for fast use
		format_strings = ','.join(['?'] * len(nocache))
		cur.execute('SELECT word, vector FROM {0} WHERE word IN ({1})'.format(table, format_strings), tuple(nocache))

		# {0} is first argument
		# {1} is second argument

		Results = [(x[0], json.loads(x[1])) for x in cur.fetchall()]
		for item in Results:
			memorybank[table][item[0]] = (item[0], item[1])
	else:
		Results = []
	if len(cache) > 0:
		for item in cache:
			Results.append(memorybank[table][item])


	'''
	The structure of memorybank:

	A dictionary of dictionaries.

	memorybank[table] -> Access the dictionary corresponding to the table of vectors denoted by 'table'

	memorybank[table][item] -> (word,vector) pairs for the year corresponding to the vector table called 'table'
	'''


	### ENDS HERE ###
	
	Cat1 = np.array([word[1] for word in Results if word[0] in X])
	Att1 = np.array([word[1] for word in Results if word[0] in A])
	Att2 = np.array([word[1] for word in Results if word[0] in B])

	#Adding in the vectors from the word lists X, A and B, for all the non-missing words

	effect_list = []#stores list of effect sizes 

	effect_size = diff_sim(Cat1,Att1,Att2)#this is the effect_size calculated with all Category words included
	
	
	for x in range(len(Cat1)):#iterate through each word in Cat1
		Cat2 = Cat1#initialize a temporary list called 'Cat2', to be equal to Cat1
		Cat2 = np.delete(Cat2,x,axis=0)#delete the 'x'th word from the category words, to calculate effect size for the list without it
		val = diff_sim(Cat2,Att1,Att2)
		effect_list.append(val)


	missing = [word for word in allwords if word not in [res[0] for res in Results]]
	result_dict = OrderedDict({
		"table": table,
		# the category labels are not fed to the function, so assign them at higher level upon return
		"categories": None,
		"attribute(s)": None,
		"effect_size": effect_size,
		"loocv_effect_sizes": effect_list,
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

	
	tablenames = ["vectors1800", "vectors1810", "vectors1820", "vectors1830", "vectors1840", "vectors1850",
	 "vectors1860", "vectors1870", "vectors1880", "vectors1890", "vectors1900", "vectors1910", 
	 "vectors1920", "vectors1930", "vectors1940", "vectors1950", "vectors1960", "vectors1970", 
	 "vectors1980", "vectors1990"]



	with open("../../python_pachankis.txt", 'r') as f:

		for line in f:
			attribute = str(line.rstrip()).split()
			
			try:

				for table in tablenames:
					
					result_dict = s_weat(attribute, vecs["good"], vecs["bad"],  # changed attribute to a "vector" here
						permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
					result_dict["categories"] = "good vs. bad"
					result_dict["attribute(s)"] = attribute[0] # here, we will want just the first item from each row of the attribute text document
					results.append(result_dict)
					

			except Exception as e:
				print(e)



	df = pd.DataFrame.from_dict(results)
	df.to_csv("loocv_results.csv", index = False, sep = '\t')
    
    
cnx.close()

print(datetime.datetime.now() - begin_time)
