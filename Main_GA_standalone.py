import pandas

from ga.GA import GA
from sys import argv
from Arguments import *
import os

from sklearn.model_selection import train_test_split

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-GA
#
# Copyright ©2021 J. E. Batista
#




def openAndSplitDatasets(which,seed):
	if VERBOSE:
		print( "> Opening: ", which )

	# Open dataset
	ds = pandas.read_csv(DATASETS_DIR+which)

	# Read header
	class_header = ds.columns[-1]

	return train_test_split(ds.drop(columns=[class_header]), ds[class_header], 
		train_size=TRAIN_FRACTION, random_state=seed, 
		stratify = ds[class_header])


def run(r,dataset):
	if VERBOSE:
		print("> Starting run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print()

	Tr_X, Te_X, Tr_Y, Te_Y = openAndSplitDatasets(dataset,r)

	# Train a model
	model = GA(POPULATION_SIZE, MAX_GENERATION, ELITISM_SIZE, MODEL_NAME, 
		THREADS, RANDOM_STATE, VERBOSE)
	model.fit(Tr_X, Tr_Y, Te_X, Te_Y)


	# Obtain training results
	accuracy  = model.getAccuracyOverTime()
	waf       = model.getWaFOverTime()
	kappa     = model.getKappaOverTime()
	size      = model.getSizeOverTime()
	model_str = model.getFeaturesOverTime()
	times     = model.getGenerationTimes()
	
	tr_acc     = accuracy[0]
	te_acc     = accuracy[1]
	tr_waf     = waf[0]
	te_waf     = waf[1]
	tr_kappa   = kappa[0]
	te_kappa   = kappa[1]

	if VERBOSE:
		print("> Ending run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print("  > Final model:", model_str[-1])
		print("  > Training accuracy:", tr_acc[-1])
		print("  > Test accuracy:", te_acc[-1])
		print()

	return (tr_acc,te_acc,
			tr_waf,te_waf,
			tr_kappa,te_kappa,
			size,
			times,
			model_str)
			

def callGA():
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:
		outputFilename = OUTPUT_DIR+"GA_"+ dataset
		if not os.path.exists(outputFilename):
			results = []

			# Run the algorithm several times
			for r in range(RUNS):
				results.append(run(r,dataset))

			# Write output header
			file = open(outputFilename , "w")
			file.write("Attribute,Run,")
			for i in range(MAX_GENERATION):
				file.write(str(i)+",")
			file.write("\n")
		
			attributes= ["Training-Accuracy","Test-Accuracy",
						 "Training-WaF", "Test-WaF",
						 "Training-Kappa", "Test-Kappa",
						 "Size", "Time",	
						 "Final_Model"]

			# Write attributes with value over time
			for ai in range(len(attributes)):
				for i in range(RUNS):	
					file.write("\n"+attributes[ai]+","+str(i)+",")
					file.write( ",".join([str(val) for val in results[i][ai]]))
				file.write("\n")

			# Write the final models
			#for i in range(len(results)):
			#	file.write("\n"+attributes[-1]+","+str(i)+",")
			#	file.write(results[i][-1])
			#file.write("\n")

			# Write some parameters
			file.write("\n\nParameters")
			file.write("\nPopulation Size,"+str(POPULATION_SIZE))
			file.write("\nMax Generation,"+str(MAX_GENERATION))
			file.write("\nModel Name,"+MODEL_NAME)
			file.write("\nElitism Size,"+str(ELITISM_SIZE))
			file.write("\nThreads,"+str(THREADS))
			file.write("\nRandom State,"+str(list(range(RUNS))))


			file.close()
		else:
			print("Filename: " + outputFilename +" already exists.")


if __name__ == '__main__':
	callGA()
