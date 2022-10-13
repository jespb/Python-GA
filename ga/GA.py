from .Individual import Individual
from .GeneticOperators import getElite, getOffspring
import multiprocessing as mp
import time
from random import Random

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GA
#
# Copyright ©2021-2022 J. E. Batista
#



class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message



class GA:
	population_size = None
	max_generation = None
	model_name = None
	elitism_size = None
	verbose = None
	threads = None
	rng = None # random number generator

	population = None
	bestIndividual = None
	actualBestIndividual = None
	currentGeneration = 0

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingWaFOverTime = None
	testWaFOverTime = None
	trainingKappaOverTime = None
	testKappaOverTime = None
	sizeOverTime = None
	featuresOverTime = None

	generationTimes = None




	def checkIfTrained(self):
		if self.population == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")


	def __init__(self, population_size = 500, max_generation = 100,	elitism_size = 1, 
		model_name = "DT", threads=1, random_state = 42, verbose = True):

		self.population_size = population_size
		self.max_generation = max_generation
		self.model_name = model_name
		self.elitism_size = elitism_size
		self.threads = max(1, threads)

		self.rng = Random(random_state)

		self.verbose = verbose


	def fit(self, Tr_x, Tr_y, Te_x, Te_y):

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y


		self.population = []

		while len(self.population) < self.population_size:
			ind = Individual(model_name = self.model_name)
			ind.create(self.rng, len(Tr_x.columns))
			self.population.append(ind)

		self.bestIndividual = self.population[0]
		self.bestIndividual.fit(self.Tr_x, self.Tr_y)
		self.actualBestIndividual = self.bestIndividual


		if not self.Te_x is None:
			self.trainingAccuracyOverTime = []
			self.testAccuracyOverTime = []
			self.trainingWaFOverTime = []
			self.testWaFOverTime = []
			self.trainingKappaOverTime = []
			self.testKappaOverTime = []
			self.sizeOverTime = []
			self.featuresOverTime = []
			self.generationTimes = []

		if self.verbose:
			print("Training a model with the following parameters: ", end="")
			print("{Population Size : "+str(self.population_size)+"}, ", end="")
			print("{Max Generation : "+str(self.max_generation)+"}, ", end="")
			print("{Model Name : "+self.model_name+"}, ", end="")
			print("{Elitism Size : "+str(self.elitism_size)+"}, ", end="")
			print("{Threads : "+str(self.threads)+"}")

		
		self.train()


	def __str__(self):
		self.checkIfTrained()
		
		return str(self.bestIndividual)


	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= self.max_generation
		perfectTraining = self.bestIndividual.getFitness() == 1
		
		return genLimit  or perfectTraining


		

	def train(self):
		'''
		Training loop for the algorithm.
		'''
		if self.verbose:
			print("> Running log:")

		while self.currentGeneration < self.max_generation:
			if not self.stoppingCriteria():
				t1 = time.time()
				self.nextGeneration()
				t2 = time.time()
				duration = t2-t1
			else:
				duration = 0
			self.currentGeneration += 1
			
			if not self.Te_x is None:
				self.trainingAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))
				self.testAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Te_x, self.Te_y, pred="Te"))
				self.trainingWaFOverTime.append(self.bestIndividual.getWaF(self.Tr_x, self.Tr_y, pred="Tr"))
				self.testWaFOverTime.append(self.bestIndividual.getWaF(self.Te_x, self.Te_y, pred="Te"))
				self.trainingKappaOverTime.append(self.bestIndividual.getKappa(self.Tr_x, self.Tr_y, pred="Tr"))
				self.testKappaOverTime.append(self.bestIndividual.getKappa(self.Te_x, self.Te_y, pred="Te"))
				self.sizeOverTime.append(self.bestIndividual.getSize())
				self.featuresOverTime.append(str(self.bestIndividual))
				self.generationTimes.append(duration)

		if self.verbose:
			print()



	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = time.time()
		
		# Calculates the accuracy of the population using multiprocessing
		if self.threads > 1:
			with mp.Pool(processes= self.threads) as pool:
				fitnesses = pool.map(fitIndividuals, [(ind, self.Tr_x, self.Tr_y) for ind in self.population] )
				for i in range(len(self.population)):
					self.population[i].fitness = fitnesses[i][0]
					self.population[i].model = fitnesses[i][1]
					self.population[i].training_X = self.Tr_x
					self.population[i].training_Y = self.Tr_y
		else:
			[ ind.fit(self.Tr_x, self.Tr_y) for ind in self.population]
			[ ind.getFitness() for ind in self.population ]

		# Sort the population from best to worse
		self.population.sort(reverse=True)


		# Update best individual
		if self.population[0] > self.actualBestIndividual:
			self.actualBestIndividual = self.population[0]
		self.bestIndividual = self.population[0]

		# Generating Next Generation
		newPopulation = []
		newPopulation.extend(getElite(self.population, self.elitism_size))
		while len(newPopulation) < self.population_size:
			offspring = getOffspring(self.rng, self.population)
			newPopulation.extend(offspring)
		self.population = newPopulation[:self.population_size]


		end = time.time()

		# Debug
		if self.verbose and self.currentGeneration%1==0:
			if not self.Te_x is None:
				print("   > Gen # %3d   Tr-Acc: %.4f  Te-Acc: %.4f   Size:%4d   Time: %.5f" % 
					(self.currentGeneration, 
						self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"), 
						self.bestIndividual.getAccuracy(self.Te_x, self.Te_y, pred="Te"), 
						self.bestIndividual.getSize() , 
						end- begin) )
			else:
				print("   > Gen #"+str(self.currentGeneration)+":  Tr-Acc: "+ "%.6f" %self.actualBestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))


	def predict(self, sample):
		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)




	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		self.checkIfTrained()

		return self.bestIndividual.predict(sample)



	def getCurrentGeneration(self):
		return self.currentGeneration


	def getBestIndividual(self):
		'''
		Returns the final M3GP model.
		'''
		self.checkIfTrained()

		return self.bestIndividual

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingAccuracyOverTime, self.testAccuracyOverTime]

	def getWaFOverTime(self):
		'''
		Returns the training and test WAF of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingWaFOverTime, self.testWaFOverTime]

	def getKappaOverTime(self):
		'''
		Returns the training and test kappa values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingKappaOverTime, self.testKappaOverTime]

	def getSizeOverTime(self):
		'''
		Returns the size values of the best model in each generation.
		'''
		self.checkIfTrained()

		return self.sizeOverTime

	def getFeaturesOverTime(self):
		'''
		Returns the size values of the best model in each generation.
		'''
		self.checkIfTrained()

		return self.featuresOverTime

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.generationTimes




def fitIndividuals(a):
	ind,x,y = a
	
	return (ind.getFitness(x,y), ind.model)