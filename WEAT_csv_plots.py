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
#pp = PdfPages('EngAll_WEAT_baseline.pdf')


#IMPORTANT NOTE!
#Be sure to change the table decade values, based on the word vectors you are using.
#i.e, EngFic (1800 - 1990) or COHA (1810 - 2000)


cnx = sqlite3.connect('vectors.decades.db')
cnx.text_factory = str

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

def weat(X, Y, A, B, permt = 0, perm_n = 10000, table = "vectors1800", cohesion_test = False, 
cohesion_permutations = 1000, cohesion_type = 2):

	def diff_sim(X, Y, A, B):

		sum_X = 0
		for x in X:
			sum_X += sim(x, A, B)

		sum_Y = 0
		for y in Y:
			sum_Y += sim(y, A, B)

		difference = sum_X/len(X) - sum_Y/len(Y)

		all_sims = []
		for w in (np.concatenate((X,Y))):
			all_sims.append(sim(w, A, B))
		# For SD calculation, assign weights based on frequency of opposite category
		weights = [len(Y) for num in range(len(X))] + [len(X) for num in range(len(Y))]
		standard_dev = weighted_std(all_sims, weights)
		effect_size = difference/standard_dev
		
		return effect_size

	def sim(w, A, B):

		w_ = w.reshape(1, -1)

		results = spatial.distance.cdist(w_, A, 'cosine')
		results_filtered = np.array([result for result in results[0] if (result == result)])
		sum_A = (1 - results_filtered).sum()

		results = spatial.distance.cdist(w_, B, 'cosine')
		results_filtered = np.array([result for result in results[0] if (result == result)])
		sum_B = (1 - results_filtered).sum()
		
		difference = sum_A/len(A) - sum_B/len(B)
		
		return difference

	def permutation_test(X, Y, A, B):
		jointlist = np.array(list(X) + list(Y))
		permutations = []
		nums = list(range(len(jointlist)))
		for comb in combinations(nums, len(X)):
			set1 = [item for i, item in enumerate(jointlist) if i in comb]
			set2 = [item for i, item in enumerate(jointlist) if i not in comb]
			permutations.append(diff_sim(set1, set2, A, B))
		return permutations

	def rand_test(X, Y, A, B, perm_n):
		jointlist = np.array(list(X) + list(Y))
		np.random.shuffle(jointlist)
		permutations = []
		count = 0
		cutpoint = len(X)
		while count < perm_n:
			np.random.shuffle(jointlist)
			set1 = jointlist[:cutpoint]
			set2 = jointlist[cutpoint:]
			permutations.append(diff_sim(set1, set2, A, B))
			count += 1
		return permutations

	allwords = list(X + Y + A + B)
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
	Cat2 = np.array([word[1] for word in Results if word[0] in Y])
	Att1 = np.array([word[1] for word in Results if word[0] in A])
	Att2 = np.array([word[1] for word in Results if word[0] in B])

	effect_size = diff_sim(Cat1, Cat2, Att1, Att2)

	if permt == 1 or permt == 2:
		if permt == 1:
			permutations = np.array(permutation_test(Cat1, Cat2, Att1, Att2))
		elif permt == 2:
			permutations = np.array(rand_test(Cat1, Cat2, Att1, Att2, perm_n = perm_n))
		perm_mean = np.mean(permutations)
		permutations = permutations - perm_mean
		sum_c = effect_size - perm_mean
		Pleft = (sum(i <= sum_c for i in permutations)+1)/(len(permutations)+1)
		Pright = (sum(i >= sum_c for i in permutations)+1)/(len(permutations)+1)
		Ptot = (sum(abs(i) >= abs(sum_c) for i in permutations)+1)/(len(permutations)+1)
		se = np.std(permutations)
	
	if cohesion_test == True:
		cohesion_categories = group_cohesion_test(Cat1, Cat2, perm_n = cohesion_permutations, permtype = cohesion_type)
		cohesion_attributes = group_cohesion_test(Att1, Att2, perm_n = cohesion_permutations, permtype = cohesion_type)

	missing = [word for word in allwords if word not in [res[0] for res in Results]]
	result_dict = OrderedDict({
		"table": table,
		"categories": None,
		"attribute(s)": None,
		"effect_size": effect_size,
		"missing": missing
	})
	if permt == 1 or permt == 2:
		result_dict["Pleft"] = Pleft
		result_dict["Pright"] = Pright
		result_dict["Ptot"] = Ptot
		result_dict["se"] = se

	if cohesion_test == True:
		result_dict["cohesion_categories"] = cohesion_categories
		result_dict["cohesion_attributes"] = cohesion_attributes

	return result_dict


def s_weat(X, A, B, permt = 0, perm_n = 10000, table = "vectors1800", cohesion_test = False, 
cohesion_permutations = 1000, cohesion_type = 2):

	def diff_sim(X, A, B):

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

	if permt == 1 or permt == 2:
		if permt == 1:
			permutations = np.array(permutation_test(Cat1, Att1, Att2))
		elif permt == 2:
			permutations = np.array(rand_test(Cat1, Att1, Att2, perm_n = perm_n))
		perm_mean = np.mean(permutations)
		permutations = permutations - perm_mean
		sum_c = effect_size - perm_mean
		Pleft = (sum(i <= sum_c for i in permutations)+1)/(len(permutations)+1)
		Pright = (sum(i >= sum_c for i in permutations)+1)/(len(permutations)+1)
		Ptot = (sum(abs(i) >= abs(sum_c) for i in permutations)+1)/(len(permutations)+1)
		se = np.std(permutations)

	if cohesion_test == True:
		cohesion_categories = group_cohesion_test(Att1, Att2, perm_n = cohesion_permutations, permtype = cohesion_type)

	missing = [word for word in allwords if word not in [res[0] for res in Results]]
	result_dict = OrderedDict({
		"table": table,
		# the category labels are not fed to the function, so assign them at higher level upon return
		"categories": None,
		"attribute(s)": None,
		"effect_size": effect_size,
		"missing": missing
	})
	if permt == 1 or permt == 2:
		result_dict["Pleft"] = Pleft
		result_dict["Pright"] = Pright
		result_dict["Ptot"] = Ptot
		result_dict["se"] = se

	if cohesion_test == True:
		result_dict["cohesion_categories"] = cohesion_categories

	return result_dict


if __name__ == '__main__':

	vecs = {
		# "female": "mother wife aunt woman girl female grandma daughter".split(),
		# "female.names": "rebecca michelle emily julia anna".split(),
		# "male": "man son father boy uncle grandpa husband male".split(), 
		# "male.names": "ben paul daniel john jeffrey".split(),
		# "home.pi": "wedding marriage parents relatives family home children".split(), 
		# "work.pi": "career corporation salary office professional management business".split(), 
		# "science.pi": "astronomy math chemistry physics biology geology engineering".split(), 
		# "arts.pi": "history arts humanities english philosophy music literature".split(), 
		# "instruments": "guitar banjo drum tuba bell fiddle harpsichord piano flute horn violin".split(), 
		# "weapons": "weapon arrow club gun spear sword dynamite rifle tank bomb knife cannon slingshot whip".split(),
		# "flower": "aster clover hyacinth marigold poppy azalea crocus iris orchid rose bluebell daffodil lilac pansy tulip buttercup daisy lily peony violet carnation gladiola magnolia petunia zinnia".split(),
		# "insect": "ant caterpillar flea locust spider bedbug centipede fly maggot tarantula bee cockroach gnat mosquito termite beetle cricket hornet moth wasp blackfly dragonfly horsefly roach weevil".split(),
		# # "good": "caress, freedom, health, love, peace, cheer, friend, heaven, loyal, pleasure, diamond, gentle, honest, lucky, rainbow, diploma, gift, honor, miracle, sunrise, family, happy, laughter, paradise, vacation".split(', '),
		# # "bad": "abuse, crash, filth, murder, sickness, accident, death, grief, poison, stink, assault, disaster, hatred, pollute, tragedy, divorce, jail, poverty, ugly, cancer, kill, rotten, vomit, agony, prison".split(', '),
		#  #"good": "good happiness happy fun fantastic lovable magical delight joy relaxing honest excited laughter lover cheerful".split(), 
		# #"bad": "bad torture murder abuse wreck die disease disaster mourning virus killer nightmare stress kill death".split(),
		# "immigrant" :"immigrant immigration detention refugee emigrant emigration emmigrate visa foreigner deportation deport quota asylum migrant checkpoint smuggling alien assimilation trafficking".split(),
		# "citizen" :"nationality patriotism citizen citizenship national residency resident residence patriotic native patriot homeland local home passport registered documented settler country nation".split(),
		# "asian" :"asian asians eastern oriental pacific mandarin cantonese chinese korean japanese vietnamese thai laotian singaporean china korea japan vietnam thailand laos".split(),
		# "white" :"white whites whitened whiteness whitewash western latin americans canadians europeans british english america usa canada europe uk england".split(),
		

		#NEW LISTS

		"instruments": "bagpipe, cello, guitar, lute, trombone, banjo, clarinet, harmonica, mandolin, trumpet, bassoon, drum, harp, oboe, tuba, bell, fiddle, harpsichord, piano, viola, bongo, flute, horn, saxophone, violin".split(', '),
		"weapons": "arrow, club, gun, missile, spear, axe, dagger, harpoon, pistol, sword, blade, dynamite, hatchet, rifle, tank, bomb, firearm, knife, shotgun, teargas, cannon, grenade, mace, slingshot, whip".split(', '),
		"black" :"dark, blacks, black, colored, africans, african, negro, race, african-american, afro-american, non-white, segregation, apartheid, afro, saharan, liberia, colonialism, kenya, nigeria, nairobi".split(', '),
		"black.names" :"washington, jefferson, booker, banks, joseph, mosley, jackson, charles, dorsey, rivers, smith, johnson, williams, brown, jones, davis, thomas".split(', '),
		#"whitenames" :"harris, nelson, robinson, thompson, moore, wright, anderson, clark, jackson, taylor, scott, davis, allen, adams, lewis, wilson, martin".split", ",
		"gay": "gay, gays, lesbian, lesbians, bisexual, lesbo, homosexual, queer, lgbt, sexuality, pride, marches, march, transgender".split(', '),
		"straight": "straight, heterosexual, heterosexuals, heterosexuality, norm, majority".split(', '),
		"light": "pale, fair, fairness, blond, light, blonde, white".split(', '),
		"dark": "black, brunette, dark, brown, dreadlock".split(', '),
		"abled": "abled, able-bodied, healthy, fit, robust, strong, sturdy, vigorous, athletic, muscular, tough, powerful, mighty, rugged, burly, brawny, buff, lusty".split(', '),
		"disabled": "disabled, handicapped, elderly, deaf, disable, unable, disabilities, crippled, wheelchair, disability, physically, disabling, mentally".split(', '),
		"old": "old, older, fashioned, rusty, familiar, elderly, mature, aged, senior, grey, senile, ancient, decrepit, ripe, venerable".split(', '),
		"young": "young, younger, teenage, boy, youngsters, girl, teenager, teenagers, teens, girls, woman, eager, talented, kid, youngster, youngest, teen, boys, guys, wonder, little, youths, naked, daughter, youth, guy, children, concerned, child, daughters, juvenile, immature".split(', '),
		"fat": "fat, fatty, weight, chubby, obese, eating, calories, fats, eat, plump, belly, diet, big, meat, saturated, overweight, diets, addiction, dieting, excess, pounds".split(', '),
		"thin": "thin, thinner, slender, soft, tiny, shape, skinny, pretty, scrawny, bony, scraggy, gaunt, slim, lean, skeletal, underfed, undernourished, slender, lanky, anorexic".split(', '),
		"male": "he his him man men himself son father husband guy boy brother male dad boyfriend king grandfather uncle grandson nephew lad gentleman fraternity bachelor prince dude gentlemen stepfather daddy sir bloke groom stepson suitor godfather grandpa fella hero fatherhood fraternities papa pa".split(),
		"female": "her hers she women woman herself daughter mother wife gal girl sister female mom girlfriend queen grandmother aunt granddaughter niece lady gentlewoman sorority bachelorette princess dame gentlewomen stepmother mommy ms madam bride stepdaughter maiden godmother grandma missus heroine motherhood sororities mama ma".split(),
		"male.names": "john, paul, mike, kevin, steve, greg, jeff, bill, noah, liam, jacob, mason, william, ethan, micheal, alexander, james, elijah, daniel, matthew, david, joseph, anthony, joshua, andrew, ben, jeffrey".split(', '),
		"female.names": "amy, joan, lisa, sarah, diana, kate, ann, donna, emma, sophia, olivia, isabella, abigail, emily, charlotte, elizabeth, chloe, sofia, grace, scarlett, lily, hannah, claire, natalie, samantha, rebecca, michelle, anna, julia".split(', '),
		"asian.names": "cho, wong, tang, huang, chu, chung, ng, wu, liu, chen, lin, yang, kim, chang, shah, wang, li, khan, singh, hong".split(', '),
		"white.names": "harris, nelson, robinson, thompson, moore, wright, anderson, clark, jackson, taylor, scott, davis, allen, adams, lewis, wilson, martin".split(', '),
		"chinese.names": "chung, liu, wong, huang, ng, hu, chu, chen, lin, liang, wang, wu, yang, tang, chang, hong, li".split(', '),
		"russian.names": "gurin, minsky, sokolov, markov, maslow, novikoff, mishkin, smirnov, orloff, ivanov, sokoloff, davidoff, savin, romanoff, babinski, sorokin, levin, pavlov, rodin, agin".split(', '),
		"hispanic.names": "ruiz, alvarez, vargas, castillo, gomez, soto, gonzalez, sanchez, rivera, mendoza, martinez, torres, rodriguez, perez, lopez, medina, diaz, garcia, castro, cruz".split(', '),
		"arab": "muslim, muslims, islamic, moslem, islam, arab, sunni, religious, islamist, moslems, arabs, mosques, religion, pakistani, secular".split(', '),
		"jew": "jewish, jews, jew, judaism, gentile, devout, zionist, synagogue, torah, bethelem, hebrew, semitic, yiddish, kosher".split(', '),
		"christian": "christian, catholic, faith, church, evangelical, religious, christians, baptist, christianity, religion, protestant, bible, lutheran, believe, theology, pastor, churches, secular, fundamentalist, biblical, orthodox, ministries, methodist, christ, belief, believers, missionary, worship, atheist, tradition, gospel, spirituality, preaching, believer, trinity, devout, grace, hope".split(', '),
		"immigrant" :"immigrant immigration detention refugee emigrant emigration emmigrate visa foreigner deportation deport quota asylum migrant checkpoint smuggling alien assimilation trafficking".split(),
		"citizen" :"nationality patriotism citizen citizenship national residency resident residence patriotic native patriot homeland local home passport registered documented settler country nation".split(),
		"black.female.names": "aisha, ebony, keisha, kenya, latonya, lakisha, latoya, tamika, tanisha, aliyah, shanice, kiara, jasmine, brianna, kayla".split(', '),
		"white.female.names": "allison, anne, carrie, emily, jill, laurie, kristen, meredith, sarah, molly, amy, claire, katie, abigail, holly".split(', '),
		"hispanic.female.names": "ana, sophia, isabella, emilia, gabriella, elena, angelina, alicia, marcia, maya, alejandra, reyna, rosa, teresa, linda".split(', '),
		"asian" :"asian asians eastern oriental pacific mandarin cantonese chinese korean japanese vietnamese thai laotian singaporean china korea japan vietnam thailand laos".split(),
		"white" :"white whites whitened whiteness whitewash western latin americans canadians europeans british english america usa canada europe uk england".split(),
		"good": "love, cheer, friend, pleasure, adore, cheerful, friendship, joyful, smiling, cherish, excellent, glad, joyous, spectacular, appealing, delight, excitement, laughing, attractive, delightful, fabulous, glorious, pleasing, beautiful, fantastic, happy, lovely, terrific, celebrate, enjoy, magnificent, triumph".split(', '),
		"bad": "abuse, grief, poison, sadness, pain, despise, failure, nasty, angry, detest, horrible, negative, ugly, dirty, gross, evil, rotten, annoy, disaster, horrific, scorn, awful, disgust, hate, humiliate, selfish, tragic, bothersome, hatred, hurtful, sickening, yucky".split(', '),
		"work.pi": "work office job business trade money manage executive professional corporate corporation salary career hire hiring".split(),
		"home.pi": "baby house home wedding kid family marry marriage parent caregiving children cousins relative kitchen cook".split(),
		"science.pi": "science scientist chemistry chemist physics doctor medicine engineer space spaceship astronaut chemical microscope technology einstein nasa experiment astronomy data analyze atom molecule lab".split(),
		"arts.pi": "art dance dancing dancer jig sing singing singer paint painting portrait sketch picture painter song draw drawing poet poetry symphony melody music musician sculpture create artistry designer compose".split(),
		"math": "puzzle number count math counting calculator subtraction addition multiplication division algebra geometry calculus equations computation computer".split(),
		"reading": "book read write story word writing reading tale novel literature narrative sentence paragraph phrase diary notebook fiction nonfiction".split(),
		"flower": "aster clover hyacinth marigold poppy azalea crocus iris orchid rose bluebell daffodil lilac pansy tulip buttercup daisy lily peony violet carnation gladiola magnolia petunia zinnia".split(),
		"insect": "ant caterpillar flea locust spider bedbug centipede fly maggot tarantula bee cockroach gnat mosquito termite beetle cricket hornet moth wasp blackfly dragonfly horsefly roach weevil".split()
	}

	s_tests = [
		# ("male", "female", "home.pi"),
		# ("male", "female", "work.pi"),
		# ("male.names", "female.names", "home.pi"),
		# ("male.names", "female.names", "work.pi"),
		# ("male", "female", "arts.pi"),
		# ("male", "female", "science.pi"),
		# ("male.names", "female.names", "arts.pi"),
		# ("male.names", "female.names", "science.pi")
		# ("immigrant","citizen","bad"),
		# ("immigrant","citizen","good"),
		# ("immigrant","citizen","asian"),
		# ("immigrant","citizen","white")

	]

	db_tests = [
		# ("male", "female", "home.pi", "work.pi"),
		# ("male.names", "female.names", "home.pi", "work.pi"),
		# ("male", "female", "arts.pi", "science.pi"),
		# ("male.names", "female.names", "arts.pi", "science.pi"),
		# ("instruments", "weapons", "good", "bad")
		# ("immigrant","citizen","bad","good"),
		# ("immigrant","citizen","asian","white")
		# ("male","female","science.pi","arts.pi"),
		# ("male","female","work.pi","home.pi"),
		# ("asian","white","bad","good")


		# ("flower", "insect", "good", "bad"),
		# ("instruments", "weapons", "good", "bad"),
		# ("white","black","good","bad"),
		# ("white.names", "black.names", "good", "bad"),
		("straight", "gay", "good", "bad"),
		("light", "dark", "good", "bad")
		# ("abled","disabled","good","bad"),
		# ("young", "old", "good", "bad"),
		# ("thin", "fat", "good", "bad"),
		# ("female", "male", "good", "bad"),
		# ("female.names","male.names","good","bad"),
		# ("white.names", "asian.names", "good", "bad"),
		# ("white.names", "chinese.names", "good", "bad"),
		# ("white.names", "russian.names", "good", "bad"),
		# ("white.names","hispanic.names","good","bad"),
		# ("jew", "arab", "good", "bad"),
		# ("christian", "arab", "good", "bad"),
		# ("citizen", "immigrant", "good", "bad"),
		# ("white.female.names","black.female.names","good","bad"),
		# ("white.female.names","hispanic.female.names","good","bad"),
		# ("male", "female", "work.pi", "home.pi"),
		# ("male", "female", "science.pi", "arts.pi"),
		# ("male", "female", "math", "reading"),
		# ("asian","white","immigrant","citizen")


	]

	results = []
	
	#change table values based on the word vectors you are analyzing. The current vectors used are EngFic

	tablenames = ["vectors1800", "vectors1810", "vectors1820", "vectors1830", "vectors1840", "vectors1850",
	 "vectors1860", "vectors1870", "vectors1880", "vectors1890", "vectors1900", "vectors1910", 
	 "vectors1920", "vectors1930", "vectors1940", "vectors1950", "vectors1960", "vectors1970", 
	 "vectors1980", "vectors1990"]


	for (att1, att2, att3, att4) in db_tests:

		try:
			# x = []
			# y = []
			for table in tablenames:

				result_dict = weat(vecs[att1], vecs[att2], vecs[att3], vecs[att4], permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
				result_dict["categories"] = att1 + " vs. " + att2
				result_dict["attribute(s)"] = att3 + " vs. " + att4
				results.append(result_dict)
				#print("Hello there")
			# 	x.append(table[-4:])
			# 	y.append(result_dict["effect_size"])


			# plt.plot(x,y)
			# plt.xlabel('Year')
			# plt.ylabel('WEAT effect size')
			# plt.title(att1+'/'+att2+' vs '+att3+'/'+att4)
			# plt.xticks(rotation = 90)
			# plt.savefig(pp, format = 'pdf')
			# plt.clf()



		except Exception as e:
			print(e)


	



	#pp.close()
		


	#print("hello")

	

	# for table in tablenames:

	# 	try:
	# 		# for (att1, att2, att3) in s_tests:
	# 		# 	result_dict = s_weat(vecs[att3], vecs[att1], vecs[att2], 
	# 		# 		permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
	# 		# 	result_dict["categories"] = att1 + " vs. " + att2
	# 		# 	result_dict["attribute(s)"] = att3
	# 		# 	results.append(result_dict)

	# 		for (att1, att2, att3, att4) in db_tests:
	# 			result_dict = weat(vecs[att1], vecs[att2], vecs[att3], vecs[att4],
	# 				permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
	# 			result_dict["categories"] = att1 + " vs. " + att2
	# 			result_dict["attribute(s)"] = att3 + " vs. " + att4
	# 			results.append(result_dict)
	# 			x.append(table[-4:])
	# 			y.append(result_dict["effect_size"])

	# 		# #Effect size for Bad/Good test

	# 		# result_dict = weat(vecs["immigrant"], vecs["citizen"], vecs["bad"], vecs["good"],
	# 		# 	permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
	# 		# result_dict["categories"] = att1 + " vs. " + att2
	# 		# result_dict["attribute(s)"] = att3 + " vs. " + att4
	# 		# results.append(result_dict)
	# 		# x1.append(table[-4:])
	# 		# y1.append(result_dict["effect_size"])




	# 		# #Effect size for Asian/White test

	# 		# result_dict = weat(vecs["immigrant"], vecs["citizen"], vecs["asian"], vecs["white"],
	# 		# 	permt=2, perm_n=1000, table = table, cohesion_test = True, cohesion_permutations = 1000, cohesion_type = 1)
	# 		# result_dict["categories"] = att1 + " vs. " + att2
	# 		# result_dict["attribute(s)"] = att3 + " vs. " + att4
	# 		# results.append(result_dict)
	# 		# x2.append(table[-4:])
	# 		# y2.append(result_dict["effect_size"])


	# 	except Exception as e:
	# 		print(e)

	df = pd.DataFrame.from_dict(results)
	df.to_csv("check.csv", index = False)




	

cnx.close()

## cd /Users/tessacharlesworth/Dropbox/1\ -\ Research\ Projects/22\ -\ Hist\ Embeddings/vectors/normalized_clean/sgns 
## python3 weat_histwords.py
