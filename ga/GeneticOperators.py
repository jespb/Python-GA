from .Individual import Individual

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GA
#
# Copyright ©2021 J. E. Batista
#

def roulette(rng, population):
  
    # Computes the totallity of the features fitness
    max = sum([indiv.getFitness() for indiv in population])

    selection = rng.random() * max

    i = -1
    while selection >= 0:
    	i += 1
    	#print( "%.5f %.5f" % (population[i].getFitness(), selection) )
    	selection -= population[i].getFitness()
    
	
    return population[i]



def getElite(population,n):
	'''
	Returns the "n" best Individuals in the population.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	return population[:n]


def getOffspring(rng, population):
	'''
	Genetic Operator: Selects a genetic operator and returns a list with the 
	offspring Individuals. The crossover GOs return two Individuals and the
	mutation GO returns one individual. Individuals over the LIMIT_DEPTH are 
	then excluded, making it possible for this method to return an empty list.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''

	p1, p2 = roulette(rng, population), roulette(rng, population)

	isCross = rng.random()<0.7
	isMut1 = rng.random()<0.5
	isMut2 = rng.random()<0.5

	if isCross <= 0.7:
		p1, p2 = GAXO(rng,p1,p2)

	if isMut1 <= 1/len(population):
		p1 = GAMUT(rng,p1)
	if isMut2 <= 1/len(population):
		p2 = GAMUT(rng,p2)

	p1.fixIndividual(rng)
	p2.fixIndividual(rng)

	return [p1,p2]



def GAXO(rng, p1, p2):
	'''
	Randomly selects one index in the feature array and creates two individuals, 
	swapping the values the arrays after the breaking point

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''

	d1 = p1.getArray()
	d2 = p2.getArray()

	cut = rng.randint(0,len(d1)-1)

	s1 = d1[:cut] + d2[cut:]
	s2 = d2[:cut] + d1[cut:]

	ret = []
	for s in [s1,s2]:
		i = Individual(model_name = p1.model_name)
		i.copy(s)
		ret.append(i)

	return ret


def GAMUT(rng, p):
	'''
	Randomly selects one position in a individual and inverts its value

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	d1 = p.getArray()

	for i in range(len(d1)):
		isFlip = rng.random() < 1/len(d1)
		d1[i] = (d1[i]+1)%2


	offspring = Individual(model_name = p.model_name)
	offspring.copy(d1)
	
	return offspring

