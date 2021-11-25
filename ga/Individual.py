from sklearn.svm import SVC

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GA
#
# Copyright ©2021 J. E. Batista
#

class Individual:
	training_X = None
	training_Y = None

	array = None

	trainingPredictions = None
	testPredictions = None
	fitness = None

	model_name = ["SVC"][0]
	model = None

	fitnessType = ["Accuracy", "WAF", "2FOLD"][0]

	def __init__(self):
		pass

	def create(self,rng, size):
		self.array = [0]*size
		for i in range(size):
			self.array[i] = rng.randint(0,1)
		
	def copy(self, array):
		self.array = array[:]



	def __gt__(self, other):
		return self.getFitness() >  other.getFitness()

	def __ge__(self, other):
		return self.getFitness() >= other.getFitness()

	def __str__(self):
		return str(self.array)


	def createModel(self):
		if self.model_name == "SVC":
			return SVC(gamma="auto", kernel="linear")

	def fit(self, Tr_x, Tr_y):
		'''
		Trains the classifier which will be used in the fitness function
		'''
		if self.model is None:
			self.training_X = Tr_x
			self.training_Y = Tr_y

			self.model = self.createModel()
	
			hyper_X = self.convert(Tr_x)

			self.model.fit(hyper_X,Tr_y)



	def getFitness(self, tr_x = None, tr_y = None):
		'''
		Returns the individual's fitness.
		'''
		if self.fitness is None:
			if not tr_x is None:
				self.training_X = tr_x
			if not tr_y is None:
				self.training_Y = tr_y


			if self.fitnessType == "Accuracy":
				self.fit(self.training_X, self.training_Y)
				self.getTrainingPredictions()
				acc = accuracy_score(self.trainingPredictions, self.training_Y)
				self.fitness = acc 

			if self.fitnessType == "WAF":
				self.fit(self.training_X, self.training_Y)
				self.getTrainingPredictions()
				waf = f1_score(self.trainingPredictions, self.training_Y, average="weighted")
				self.fitness = waf 

			if self.fitnessType == "2FOLD":
				hyper_X = self.convert(self.training_X)

				X1 = hyper_X.iloc[:len(hyper_X)//2]
				Y1 = self.training_Y[:len(self.training_Y)//2]
				X2 = hyper_X.iloc[len(hyper_X)//2:]
				Y2 = self.training_Y[len(self.training_Y)//2:]

				M1 = self.createModel()
				M1.fit(X1,Y1)
				P1 = M1.predict(X2)

				M2 = self.createModel()
				M2.fit(X2,Y2)
				P2 = M2.predict(X1)

				f1 = accuracy_score(P1, Y2)
				f2 = accuracy_score(P2, Y1)
				self.fitness = (f1+f2)/2

		return self.fitness


	def getTrainingPredictions(self):
		if self.trainingPredictions is None:
			self.trainingPredictions = self.predict(self.training_X)

		return self.trainingPredictions

	def getTestPredictions(self, X):
		if self.testPredictions is None:
			self.testPredictions = self.predict(X)

		return self.testPredictions

	def getArray(self):
		return self.array[:]

	
	def getAccuracy(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return accuracy_score(pred, Y)


	def getWaF(self, X, Y,pred=None):
		'''
		Returns the individual's WAF.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return f1_score(pred, Y, average="weighted")


	def getKappa(self, X, Y,pred=None):
		'''
		Returns the individual's kappa value.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return cohen_kappa_score(pred, Y)



	def convert(self, X):
		'''
		Returns the converted input space.
		'''
		
		cols = []
		for i in range(len(self.array)):
			if self.array[i] == 1:
				cols.append(X.columns[i])

		return X[cols]


	def predict(self, X):
		'''
		Returns the class prediction of a sample.
		'''
		hyper_X = self.convert(X)
		predictions = self.model.predict(hyper_X)

		return predictions


