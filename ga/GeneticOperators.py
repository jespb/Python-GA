from .Individual import Individual

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GA
#
# Copyright ©2021 J. E. Batista
#


def tournament(rng, population,n):
	'''
	Selects "n" Individuals from the population and return a 
	single Individual.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	candidates = [rng.randint(0,len(population)-1) for i in range(n)]
	return population[min(candidates)]


def getElite(population,n):
	'''
	Returns the "n" best Individuals in the population.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	return population[:n]


def getOffspring(rng, population, tournament_size):
	'''
	Genetic Operator: Selects a genetic operator and returns a list with the 
	offspring Individuals. The crossover GOs return two Individuals and the
	mutation GO returns one individual. Individuals over the LIMIT_DEPTH are 
	then excluded, making it possible for this method to return an empty list.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	isCross = rng.random()<0.5
	desc = None

	if isCross:
		desc = GAXO(rng, population, tournament_size)
	else:
		desc = GAMUT(rng, population, tournament_size)
	
	return desc



def GAXO(rng, population, tournament_size):
	'''
	Randomly selects one index in the feature array and creates two individuals, 
	swapping the values the arrays after the breaking point

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = tournament(rng, population, tournament_size)
	ind2 = tournament(rng, population, tournament_size)

	d1 = ind1.getArray()
	d2 = ind2.getArray()

	cut = rng.randint(0,len(d1)-1)

	s1 = d1[:cut] + d2[cut:]
	s2 = d2[:cut] + d1[cut:]

	ret = []
	for s in [s1,s2]:
		i = Individual()
		i.copy(s)
		ret.append(i)

	return ret


def GAMUT(rng, population, tournament_size):
	'''
	Randomly selects one position in a individual and inverts its value

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = tournament(rng, population, tournament_size)

	d1 = ind1.getArray()

	select = rng.randint(0,len(d1)-1)

	d1[select] = (d1[select]+1)%2

	ret = []
	for s in [d1]:
		i = Individual()
		i.copy(s)
		ret.append(i)

	return ret

