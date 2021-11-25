from .Population import Population

from random import Random

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GA
#
# Copyright ©2021 J. E. Batista
#

class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message


class GA:
	population = None

	population_size = None
	max_generation = None
	tournament_size = None
	elitism_size = None
	threads = None
	verbose = None

	rng = None # random number generator

	def checkIfTrained(self):
		if self.population == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")


	def __init__(self, population_size = 500, max_generation = 100, tournament_size = 5, 
		elitism_size = 1, threads=1, random_state = 42, verbose = True):

		self.population_size = population_size
		self.max_generation = max_generation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size
		self.threads = max(1, threads)

		self.rng = Random(random_state)

		self.verbose = verbose


	def __str__(self):
		self.checkIfTrained()
		
		return str(self.getBestIndividual())
		

	def fit(self,Tr_X, Tr_Y, Te_X = None, Te_Y = None):
		if self.verbose:
			print("Training a model with the following parameters: ", end="")
			print("{Population Size : "+str(self.population_size)+"}, ", end="")
			print("{Max Generation : "+str(self.max_generation)+"}, ", end="")
			print("{Tournament Size : "+str(self.tournament_size)+"}, ", end="")
			print("{Elitism Size : "+str(self.elitism_size)+"}, ", end="")
			print("{Threads : "+str(self.threads)+"}")

		self.population = Population(Tr_X, Tr_Y, Te_X, Te_Y, self.population_size, 
			self.max_generation, self.tournament_size, self.elitism_size, self.threads, 
			self.rng, self.verbose)
		
		self.population.train()

	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		self.checkIfTrained()

		return self.population.getBestIndividual().predict(dataset)

	def getBestIndividual(self):
		'''
		Returns the final M3GP model.
		'''
		self.checkIfTrained()

		return self.population.getBestIndividual()

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.population.getTrainingAccuracyOverTime(), self.population.getTestAccuracyOverTime()]

	def getWaFOverTime(self):
		'''
		Returns the training and test WAF of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.population.getTrainingWaFOverTime(), self.population.getTestWaFOverTime()]

	def getKappaOverTime(self):
		'''
		Returns the training and test kappa values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.population.getTrainingKappaOverTime(), self.population.getTestKappaOverTime()]

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.population.getGenerationTimes()