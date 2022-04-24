import os
from re import A
import time
import numpy as np
from numpy.random import f
import pandas as pd
import scipy

from sklearn.model_selection import KFold, StratifiedKFold
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds, fsolve, brentq
from scipy.stats import mvn, wishart, invwishart, gaussian_kde
from itertools import chain, combinations, permutations

from datetime import datetime
import pickle

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sendEmail import *

def runHTVSP():
	isQuick = False
	prinprintOutput = True
	screeningApproach = "" #"hard"
	regressor = ""

	# algorithms = ['proposedLambda', 'baseline']
	algorithms = ['proposedLambda']
	lambdaLast = np.array([2.5, 3.2])
	# lambdaLast = np.array([2.5])
	models = np.array(['ML1', 'ML2', 'ML3', 'ML4', 'ML5', 'RP (V) - DFT'])
	if len(regressor) == 0:
		df = pd.read_excel("../01_Dataset/postprocessed_RS.xlsx", sheet_name="Train")
	else:
		features = [['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'RP (V) - DFT'],
					['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV) (est)', 'LUMO (eV) (est)', 'Band Gap (est)', 'RP (V) - DFT'],
					['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'RP (V) - DFT'],
					['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'EA (eV) (est)', 'RP (V) - DFT'],
					['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap', 'EA (eV)', 'RP (V) - DFT']]
		feature_causality = {'HOMO (eV) (est)': ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings'],
							'LUMO (eV) (est)': ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings'],
							'Band Gap (est)': ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings'],
							'EA (eV) (est)': ['# C', '# B', '# O', '# Li', '# H', 'No. of Aromatic Rings', 'HOMO (eV)', 'LUMO (eV)', 'Band Gap']}
		df = pd.read_excel("../01_Dataset/preprocessed_RS.xlsx", sheet_name="Train")
	if len(lambdaLast) == 1:
		df_remaining = df.iloc[2: 6]
		df = df.drop(index=[2,3,4,5])
	else:
		df_remaining = df.iloc[: 4]
		df = df.iloc[4:]
	
	if len(lambdaLast) == 1:
		Y = df['RP (V) - DFT'] >= lambdaLast[0]
	else:
		Y = (df['RP (V) - DFT'] >= np.min(lambdaLast)) & (df['RP (V) - DFT'] <= np.max(lambdaLast))
	
	kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
	index_dataset = 0
	for test_index, train_index in kf.split(df, Y):
		if len(regressor) == 0:
			df_train, df_test = pd.concat([df.iloc[train_index], df_remaining]).copy(), df.iloc[test_index].copy()
		else:
			df_train, df_test = evaluateSamples(regressor, pd.concat([df.iloc[train_index], df_remaining]).copy(), df.iloc[test_index].copy(), features, feature_causality)
		evaluateOCC(lambdaLast, models, index_dataset, df.iloc[train_index], df.iloc[test_index], isQuick, algorithms, "EM", prinprintOutput, screeningApproach)
		index_dataset += 1
	import socket
	sendEmail(socket.gethostname() + " - Simulation END!!!!!")

def evaluateSamples(regressor, df_training, df_test, features, feature_causality):
	df_train_new = pd.DataFrame(index = df_training.index)
	df_test_new = pd.DataFrame(index = df_test.index)

	for i, feature in enumerate(features):
		for j, feat in enumerate(feature):
			if '(est)' in feat:
				feature_cols = feature_causality[feat]
				feature_out = feat.replace(" (est)", "")

				#splitting into dependant and independant variables
				X_train = df_training.loc[:, feature_cols]
				y_train = df_training[feature_out]

				X_test = df_test.loc[:, feature_cols]
				y_test = df_test[feature_out]

				#normalizing 
				scaler = StandardScaler()  
				scaler.fit(X_train)  
				X_train = scaler.transform(X_train)  
				X_test = scaler.transform(X_test)  

				if regressor == "MLP":
					tuned_parameters = {"hidden_layer_sizes": [(50,),(100,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.0001,0.05], 'learning_rate': ['constant', 'adaptive'],}

					clf = GridSearchCV(MLPRegressor(), tuned_parameters, n_jobs= 4, cv=5)
				elif regressor == "SVR":
					tuned_parameters = {'kernel' : ('poly', 'rbf', 'sigmoid'), 'C' : [1,5,10], 'degree' : [3,8], 'coef0' : [0.01,10,0.5], 'gamma' : ('auto','scale'),}

					clf = GridSearchCV(SVR(), tuned_parameters, n_jobs= 2, cv=5)
				else:
					tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.01]}]

				clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5)

				clf.fit(X_train, y_train)
				df_training[feat] = clf.predict(X_train)
				df_test[feat] = clf.predict(X_test)

		feature_cols = feature[:-1]
		feature_out = feature[-1]

		#splitting into dependant and independant variables
		X_train = df_training.loc[:, feature_cols]
		y_train = df_training[feature_out]

		X_test = df_test.loc[:, feature_cols]
		y_test = df_test[feature_out]

		#normalizing 
		scaler = StandardScaler()  
		scaler.fit(X_train)  
		X_train = scaler.transform(X_train)  
		X_test = scaler.transform(X_test)  

		if regressor == "MLP":
			tuned_parameters = {"hidden_layer_sizes": [(50,),(100,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.0001,0.05], 'learning_rate': ['constant', 'adaptive'],}

			clf = GridSearchCV(MLPRegressor(), tuned_parameters, n_jobs= 4, cv=5)
		elif regressor == "SVR":
			tuned_parameters = {'kernel' : ('poly', 'rbf', 'sigmoid'), 'C' : [1,5,10], 'degree' : [3,8], 'coef0' : [0.01,10,0.5], 'gamma' : ('auto','scale'),}

			clf = GridSearchCV(SVR(), tuned_parameters, n_jobs= 2, cv=5)
		else:
			tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.01]}]
			#tuned_parameters = [{'kernel':["linear","rbf"],'alpha': np.logspace(-3,0,100)}]

		clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5)

		clf.fit(X_train, y_train)

		df_train_new['ML' + str(i+1)] = clf.predict(X_train)
		df_test_new['ML' + str(i+1)] = clf.predict(X_test)

	df_train_new['RP (V) - DFT'] = df_training['RP (V) - DFT']
	df_test_new['RP (V) - DFT'] = df_test['RP (V) - DFT']
		
	return df_train_new, df_test_new

def evaluateOCC(lambdaLast, models, index_dataset, df_training, df_test, isQuick, algorithms, densityEstimation = "MLE", prinprintOutput = False, screeningApproach = "hard"):
	alpha = 0.5
	defaultCost = np.array([7, 4, 32949, 2, 33991, 7890])

	X0 = len(df_test)
	if len(lambdaLast) == 1:
		GT = len(df_test[df_test[models[-1]] >= lambdaLast[0]])
	else:
		GT = len(df_test[(df_test[models[-1]] >= np.min(lambdaLast)) & (df_test[models[-1]] <= np.max(lambdaLast))])

	# stageOrder = list(range(len(models)))
	# allStageSetups = list(chain.from_iterable(combinations(stageOrder, r) for r in range(len(stageOrder)+1)))
	# stageSetups = list()
	# for i in range(len(allStageSetups)):
	# 	stageSetup = allStageSetups[i]
	# 	if (len(stageSetup) > 1) and (stageSetup[-1] == stageOrder[-1]):
	# 		if len(stageSetup) > 2:
	# 			allPermutations = list(permutations(stageSetup))
	# 			for permutation in allPermutations:
	# 				if (permutation[-1] == stageOrder[-1]):
	# 					stageSetups.append(permutation)
	# 		else:
	# 			stageSetups.append(stageSetup)
	stageSetups = [(0, 1, 2, 3, 4, 5)]
	data = {}
	data['costPerStage'] = defaultCost
	data['numberOfSamples'] = X0
	data['GT'] = GT
	i = 0
	for stageSetup in stageSetups: 
		stageSetup = np.array(stageSetup)

		f_s = estimateScoreDensity(densityEstimation, df_training[models[stageSetup]])
		
		reorderedC = np.array(defaultCost)
		reorderedC = reorderedC[stageSetup]

		listOutput = list() 
		dicOutput = runAlgorithms(isQuick, df_test, models[stageSetup], algorithms, f_s, X0, reorderedC, lambdaLast, alpha, GT, True, stageSetup, prinprintOutput, screeningApproach)
		listOutput.append(dicOutput)
		outAlgorithms = dicOutput['algorithms']
		standardMethod = 0
		for i in range(len(outAlgorithms)):
			if 'proposed' in outAlgorithms[i]:
				standardMethod = i
				break
		data[''.join(map(str, stageSetup))] = listOutput

	if len(lambdaLast) == 1:
		resultPath = './results/data' + "".join(map(str, np.int0(defaultCost))) + '/'
	else:
		resultPath = './results/data' + "".join(map(str, np.int0(defaultCost))) + 'r/'

	if not os.path.isdir(resultPath):
		os.mkdir(resultPath) 
	with open(resultPath +  'CV' + str(index_dataset) + "_" + str(alpha) + '_f2.pickle', 'wb') as handle:
		pickle.dump(data, handle)
	# sendEmail('Done!')

def estimateScoreDensity(method, df_data):
	score_density = dict()
	score_density["method"] = method

	if method == "KDE":
		score_density["kernel"] = gaussian_kde(df_data.transpose().values, bw_method = 0.2)
		score_density["max"] = df_data.values.max(axis = 0)
		score_density["min"] = df_data.values.min(axis = 0)
	else:
		df_n = df_data
		df_p = df_data
		if method == "MLE":
			# GMM - MLE
			pi_n = len(df_n)/len(df_data)
			mu_n = np.mean(df_n.values, axis=0)
			sigma_n = np.cov(df_n.values.T, bias=False)  

			pi_p = len(df_p)/len(df_data)
			mu_p = np.mean(df_p.values, axis=0)
			sigma_p = np.cov(df_p.values.T, bias=False) 

			score_density['pi'] = np.array([pi_n, pi_p])
			score_density['mu'] = np.array([mu_n, mu_p])
			score_density['sigma'] = np.array([sigma_n, sigma_p])

		elif method == "EM":
			# GMM - EM
			x_train	= df_data.values

			# fit a Gaussian Mixture Model with two components
			modelDistribution = GaussianMixture(n_components=2, covariance_type='full').fit(x_train)

			score_density['pi'] = modelDistribution.weights_
			score_density['mu'] = modelDistribution.means_
			score_density['sigma'] = modelDistribution.covariances_

	return score_density

def runAlgorithms(isQuick, df, metrics, algorithms, f_s = dict({"method": "invalid"}),
			X0 = 10**5, c = [1,10,100,1000], lambdaLast = 0, alpha = 0.5, GT = 0, sampleEvaluation=False, stageSetup = 0, printOutput = False, screeningApproach = "hard"):

	assert(f_s["method"] != "invalid")

	if f_s["method"] == "EM" or f_s["method"] == "MLE":
		pi = f_s['pi']
		mu = f_s['mu']
		sigma = f_s['sigma']

	numberOfStages = len(c)
	violationTolerance  = 0.01

	dicOutput = {}
	dicOutput['algorithms'] = algorithms

	print("Number of true candidates: {:.0f}".format(GT))
	lambdas = np.concatenate((np.asarray([0]*len(lambdaLast)*(numberOfStages-1)), lambdaLast))
	lambdas = np.concatenate((lambdas[0: : 2], lambdas[1: : 2])) # [lower bounds, upper bounds]

	operator_temp, dicOutput[algorithms[0] + '_cStar'], dicOutput[algorithms[0] + '_time'] = min_f_c(isQuick, f_s, lambdaLast, c, alpha)
	if len(lambdaLast) == 1:
		lambdas = np.concatenate((operator_temp, lambdaLast))
	else:
		lambdas = np.concatenate((operator_temp[: len(c) - 1], [lambdaLast[0]], operator_temp[len(c) - 1:], [lambdaLast[1]]))
	_, samples = validateOperatorAnalytically(lambdas, f_s, c, X0, violationTolerance, algorithm=algorithms[0], printOutput = printOutput)
	# dicOutput[algorithms[0] + '_cost'], dicOutput[algorithms[0] + '_samples'], dicOutput[algorithms[0] + 'sensitivity'], dicOutput[algorithms[0] + 'specificity'], dicOutput[algorithms[0] + 'F1'], dicOutput[algorithms[0] + 'accuracy'], dicOutput['sensitivity'], dicOutput['specificity'], dicOutput['F1'], dicOutput['accuracy'] = validateOperatorBySamples(df.copy(), metrics, lambdas, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm=algorithms[0], printOutput = printOutput)
	dicOutput[algorithms[0] + '_cost'], dicOutput[algorithms[0] + '_samples'], dicOutput[algorithms[0] + 'sensitivity'], dicOutput[algorithms[0] + 'specificity'], dicOutput[algorithms[0] + 'F1'], dicOutput[algorithms[0] + 'accuracy'] = validateOperatorBySamples(df.copy(), metrics, lambdas, samples, f_s, c, X0, violationTolerance, stageSetup, algorithm=algorithms[0], printOutput = printOutput, screeningApproach = screeningApproach)
	dicOutput[algorithms[0] + '_lambdas'] = lambdas
	dicOutput[algorithms[0] + '_totalCost'] = sum(dicOutput[algorithms[0] + '_cost'])

	# lambdas = [0]*(numberOfStages-1) + [lambdaLast]
	# # lambdas[:-1], dicOutput[algorithms[1] + '_cStar'], dicOutput[algorithms[1] + '_time'] = baseApproach2(pi, mu, sigma, lambdas[-1], sum(dicOutput[algorithms[0] + '_cost']), c, X0, violationTolerance)
	# lambdas[:-1], dicOutput[algorithms[1] + '_decayRatio'], dicOutput[algorithms[1] + '_time'], samples = baseApproachMine(df.copy(), metrics, dicOutput[algorithms[0] + '_samples'][-1], stageSetup, pi, mu, sigma, lambdas[-1], sum(dicOutput[algorithms[0] + '_cost']), c, X0, violationTolerance)
	# # _, samples = validateOperatorAnalytically(lambdas, pi, mu, sigma, c, X0, violationTolerance, algorithm=algorithms[1], printOutput = printOutput)
	# # dicOutput[algorithms[1] + '_cost'], dicOutput[algorithms[1] + '_samples'], dicOutput[algorithms[1] + 'sensitivity'], dicOutput[algorithms[1] + 'specificity'], dicOutput[algorithms[1] + 'F1'], dicOutput[algorithms[1] + 'accuracy'], _, _, _, _ = validateOperatorBySamples(df.copy(), metrics, lambdas, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm=algorithms[1], printOutput = printOutput)
	# dicOutput[algorithms[1] + '_cost'], dicOutput[algorithms[1] + '_samples'], dicOutput[algorithms[1] + 'sensitivity'], dicOutput[algorithms[1] + 'specificity'], dicOutput[algorithms[1] + 'F1'], dicOutput[algorithms[1] + 'accuracy'], _, _, _, _ = validateOperatorBySamples(df.copy(), metrics, lambdas, samples, pi, mu, sigma, c, X0, violationTolerance, stageSetup, algorithm=algorithms[1], printOutput = printOutput)
	# dicOutput[algorithms[1] + '_lambdas'] = lambdas
	# dicOutput[algorithms[1] + '_totalCost'] = sum(dicOutput[algorithms[1] + '_cost'])
	return dicOutput

def initialFunction(x, pi, mu, sigma, C_total, c, X0, target):
	lowerBound = [-np.Inf]*(len(mu[0])-1) + [x[0]]
	AUC = (pi[0]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[0]), mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[1]), mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	return (target- (AUC*X0))

def min_f_c(isQuick, f_s, lambdaLast, c, alpha):
	p = problem(f_s, lambdaLast, c, alpha)
	if len(lambdaLast) == 2:
		lc = NonlinearConstraint(p.constraintFunction, [-np.inf]*(len(c)-1), [0]*(len(c)-1), keep_feasible = True)

	if f_s["method"] == "EM" or f_s["method"] == "MLE":
		mu = f_s['mu']
		sigma = f_s['sigma']

		SD_0 = np.sqrt(np.diag(sigma[0]))
		lower_0 = mu[0][:-1] - (8*SD_0[:-1])
		upper_0 = mu[0][:-1] + (8*SD_0[:-1])
		SD_1 = np.sqrt(np.diag(sigma[1]))
		lower_1 = mu[1][:-1] - (8*SD_1[:-1])
		upper_1 = mu[1][:-1] + (8*SD_1[:-1])
	else:
		lower_0 = f_s["min"][:-1]
		lower_1 = f_s["min"][:-1]
		upper_0 = f_s["max"][:-1]
		upper_1 = f_s["max"][:-1]

	if len(lambdaLast) == 1:
		bounds = Bounds(np.min(np.vstack([lower_0, lower_1]), axis = 0), np.max(np.vstack([upper_0, upper_1]), axis = 0))
	else:
		minVec = np.min(np.vstack([lower_0, lower_1]), axis = 0)
		minVec = np.concatenate((minVec, minVec))
		maxVec = np.max(np.vstack([upper_0, upper_1]), axis = 0)
		maxVec = np.concatenate((maxVec, maxVec))
		bounds = Bounds(minVec, maxVec)

	start = time.time()
	if isQuick:
		result = differential_evolution(p.objectiveFunction, bounds)
	else:
		if len(lambdaLast) == 1:
			result = differential_evolution(p.objectiveFunction, bounds, tol=10**(-30), mutation=[0.01, 1], workers = 20, maxiter=10000, popsize=20)
		else:
			# for 0.5: result = differential_evolution(p.objectiveFunction, bounds, constraints=lc, tol=10**(-30), mutation=[0.01, 1], workers = 20, maxiter=10000, popsize=20)
			result = differential_evolution(p.objectiveFunction, bounds, constraints=lc, tol=10**(-30), mutation=[0.005, 1], workers = 20, maxiter=20000, popsize=20)
	end = time.time()
	print(result.fun)

	return result.x, result.fun, (end - start)

class problem:
	def __init__(self, f_s, lambdaLast, c, alpha):
		# Score distribution
		self.method = f_s["method"]
		if self.method == "EM" or self.method == "MLE":
			self.pi = f_s["pi"]
			self.mu = f_s["mu"]
			self.sigma = f_s["sigma"]
		elif self.method == "KDE":
			self.f_s = f_s["kernel"]
		# Given threshold
		self.lambdaLast = lambdaLast
		# Simulation cost
		self.alpha = alpha
		self.c = c

	def objectiveFunction(self, integrationBound):
		LB = [-np.Inf]*len(self.c)
		UB = [np.Inf]*len(self.c)

		f = self.c[0]
		if self.method == "EM" or self.method == "MLE":
			if len(self.lambdaLast) == 1:
				for i in range(len(self.c) - 1):
					LB[i] = integrationBound[i]
					AUC = (self.pi[0]*mvn.mvnun(LB, [np.Inf]*len(self.mu[0]), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(LB, [np.Inf]*len(self.mu[0]), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
					f += (self.c[i+1]*AUC)
				c1 = (self.pi[0]*mvn.mvnun(np.append([-np.Inf]*len(integrationBound), self.lambdaLast), [np.Inf]*len(self.mu[0]), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) 
				c1 += (self.pi[1]*mvn.mvnun(np.append([-np.Inf]*len(integrationBound), self.lambdaLast), [np.Inf]*len(self.mu[1]), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
				c2 = (self.pi[0]*mvn.mvnun(np.append(integrationBound, self.lambdaLast), [np.Inf]*len(self.mu[0]), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) 
				c2 += (self.pi[1]*mvn.mvnun(np.append(integrationBound, self.lambdaLast), [np.Inf]*len(self.mu[1]), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
				c = ((c1 - c2)/c1)
			else:
				for i in range(len(self.c) - 1):
					LB[i] = integrationBound[i]
					UB[i] = integrationBound[i + len(self.c)-1]
					AUC = (self.pi[0]*mvn.mvnun(LB, UB, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(LB, UB, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
					f += (self.c[i+1]*AUC)
				c1 = (self.pi[0]*mvn.mvnun(np.append([-np.Inf]*(len(self.c)-1), self.lambdaLast.min()), np.append([np.Inf]*(len(self.c)-1), self.lambdaLast.max()), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) 
				c1 += (self.pi[1]*mvn.mvnun(np.append([-np.Inf]*(len(self.c)-1), self.lambdaLast.min()), np.append([np.Inf]*(len(self.c)-1), self.lambdaLast.max()), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
				c2 = (self.pi[0]*mvn.mvnun(np.append(integrationBound[:len(self.c)-1], self.lambdaLast.min()), np.append(integrationBound[len(self.c)-1:], self.lambdaLast.max()), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) 
				c2 += (self.pi[1]*mvn.mvnun(np.append(integrationBound[:len(self.c)-1], self.lambdaLast.min()), np.append(integrationBound[len(self.c)-1:], self.lambdaLast.max()), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
				c = ((c1 - c2)/c1)
		else:
			for i in range(len(self.c) - 1):
				LB[i] = integrationBound[i]
				UB[i] = integrationBound[i + len(self.c)-1]
				AUC = self.f_s.integrate_box(LB, UB, maxpts=len(self.c)*10000) 
				f += (self.c[i+1]*AUC)
			c1 = self.f_s.integrate_box(np.append([-np.Inf]*(len(self.c)-1), self.lambdaLast.min()), np.append([np.Inf]*(len(self.c)-1), self.lambdaLast.max()), maxpts=len(self.c)*10000)
			c2 = self.f_s.integrate_box(np.append(integrationBound[:len(self.c)-1], self.lambdaLast.min()), np.append(integrationBound[len(self.c)-1:], self.lambdaLast.max()), maxpts=len(self.c)*10000)
			c = ((c1 - c2)/c1)
		return ( ( (1 - self.alpha) * ( f/(len(self.c)*np.max(self.c)) ) ) + (self.alpha * c) )

	def constraintFunction(self, integrationBound):
		constraintValue = integrationBound[:len(self.c)-1] - integrationBound[len(self.c)-1:] 
		return np.array(constraintValue)

def integrateCost(pi, mu, sigma, lowerBound, upperBound):
	return (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])

def baseApproachMine(dfOriginal, metrics, targetSamples, stageSetup, pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	start = time.time()
	# decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, dfOriginal, metrics, lambdaLast, c, X0))
	decayRatio = 0.75 # pass 30 % of samples
	lambdas = [-np.Inf]*(len(mu[0])-1)
	reorderedTable = dfOriginal[metrics]
	numberOfPassedSamples = [X0] + [0]*(len(metrics))
	for i in range(len(metrics)):
		if i < (len(metrics)-1):
			oneAddreorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)) + 1, metrics[i])
			reorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)), metrics[i])
			lambdas[i] = (np.min(reorderedTable[metrics[i]]) + np.min(oneAddreorderedTable[metrics[i]]))/2
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[metrics[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)
	end = time.time()
	# if targetSamples != numberOfPassedSamples[-1]:
	# 	sendEmail("Baseline works incorrectly targetSamples: {}, numberOfPassedSamples: {}".format(targetSamples, numberOfPassedSamples[-1]))
	return lambdas, decayRatio, (end - start), numberOfPassedSamples

def findDecayRatio(r, targetSamples, df, metrics, lambdaLast, c, X0):
	reorderedTable = df[metrics]
	computationalCostVector = [0]*(len(metrics))
	numberOfPassedSamples = [X0] + [0]*(len(metrics))
	for i in range(len(metrics)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		
		if i < (len(metrics)-1):
			reorderedTable = reorderedTable.nlargest(int(r*len(reorderedTable)), metrics[i])
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[metrics[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)

	return (targetSamples - numberOfPassedSamples[-1])

def baselineFindLambdas(x, i, LB, pi, mu, sigma, lambdaLast, c, X0, decayRatio):
	LB[i] = x[0]
	Xi = integrateCost(pi, mu, sigma, LB + [-np.Inf], [np.Inf]*len(mu[0]))
	return (Xi - (decayRatio**(i+1)))

def baseApproach2(pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	start = time.time()
	lambdas = [-np.Inf]*(len(mu[0])-1)
	costAvailable = C_total
	for i in range(len(lambdas)):
		isSolutionProper = False
		initialPoint = 0
		numberOfTrial = 0
		costAvailable -= (c[i] * X0 * integrateCost(pi, mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu[0])))
		costAvailableNextStage = costAvailable/ (len(mu[0])-(i+1))
		if costAvailableNextStage >= ( c[i+1]*X0*integrateCost(pi, mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu[0]))):
			lambdas[i] = -np.Inf
		else:
			while not isSolutionProper:
				lambdas[i] = fsolve(baselineFunction2, initialPoint, args=(i, lambdas, pi, mu, sigma, lambdaLast, C_total, c, X0, costAvailableNextStage), maxfev=1000000)[0]
				costExpected = c[i+1]*X0*integrateCost(pi, mu, sigma, lambdas + [-np.Inf], [np.Inf]*len(mu[0]))
				if (np.abs(costAvailableNextStage - costExpected) < (costAvailableNextStage*0.001)) and ((costAvailableNextStage*1.001) >= costExpected):
					isSolutionProper = True
				else:
					numberOfTrial += 1
					if ((numberOfTrial%5000) == 0) and (numberOfTrial > 50):
						print("No solution was found retry: " + str(numberOfTrial))
					initialPoint = np.random.uniform(-20, 20, 1)
					if numberOfTrial > 10000:
						sendEmail("No solution was found retry: " + str(costExpected))
						break
	end = time.time()
	f_star = 1 - integrateCost(pi, mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu[0]))
	return lambdas, f_star, (end - start)

def baselineFunction2(x, i, LB, pi, mu, sigma, lambdaLast, C_total, c, X0, availableCost):
	LB[i] = x[0]
	AUC = (pi[0] * mvn.mvnun(LB + [-np.Inf], [np.Inf]*len(mu[0]), mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1] * mvn.mvnun(LB + [-np.Inf], [np.Inf]*len(mu[1]), mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	Xi = X0*AUC
	return (availableCost - (c[i+1]*Xi))

def validateOperatorAnalytically(x, f_s, c, X0, violationTolerance, algorithm = "unknown", printOutput = True):
	print(algorithm + ": {}".format(x))

	method = f_s["method"]
	if method == "EM" or method == "MLE":
		pi = f_s["pi"]
		mu = f_s["mu"]
		sigma = f_s["sigma"]
	elif method == "KDE":
		kernel = f_s["kernel"]

	upperBound = [np.Inf]*len(c)
	lowerBound = [-np.Inf]*len(c)
	computationalCostVector = [0]*len(c)
	numberOfPassedSamples = [X0] + [0]*len(c)

	for i in range(len(c)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		lowerBound[i] = x[i]
		if len(x) != len(c):
			upperBound[i] = x[i + len(c)]

		if method == "EM" or method == "MLE":
			AUC = (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + ((pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0]))
		else:
			AUC = kernel.integrate_box(lowerBound, upperBound, maxpts=len(c)*10000)

		numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*AUC
		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))
	
	if method == "EM" or method == "MLE":
		AUC = (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	else:
		AUC = kernel.integrate_box(lowerBound, upperBound, maxpts=len(c)*10000)
	
	f_star = (1 - AUC)
	# if C_total < sum(computationalCostVector):
	# 	if np.abs(C_total - sum(computationalCostVector)) > (C_total*violationTolerance):
	# 		print("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	# 		sendEmail("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples

def validateOperatorBySamples(df, models, x, samples, f_s, c, X0, violationTolerance, stageSetup, algorithm = "unknown", printOutput = True, screeningApproach = "hard"):
	df_positive = df.copy()

	computationalCostVector = [0]*(len(c))
	numberOfPassedSamples = [X0] + [0]*(len(c))

	list_sensitivity = list()
	list_specificity = list()
	list_F1 = list()
	list_accuracy = list()

	for i in range(len(c)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		if len(x) == len(c):
			if i < (len(x)-1):
				df_positive = df_positive.nlargest(int(samples[i+1]), models[i])
			else:
				df_positive = df_positive[df_positive[models[i]] > x[-1]]
		else:
			df_positive = df_positive[(df_positive[models[i]] >= x[i]) & (df_positive[models[i]] <= x[i + len(c)])]
			if screeningApproach != "hard":
				if i < (len(c)-1):
					if len(df_positive) > samples[i + 1]:
						df_positive = df_positive.nlargest(int(samples[i + 1]), models[i])

		numberOfPassedSamples[i+1] = len(df_positive)
		if printOutput:
			print("\tStage{}(cost:{:.0f}*samples:{:.2f}={:.2f}) - samples passed:{:.2f}".format(i+1, c[i], numberOfPassedSamples[i], c[i]* numberOfPassedSamples[i], numberOfPassedSamples[i+1]))

		dfNetagive = pd.concat([df, df_positive, df_positive]).drop_duplicates(keep=False)
		if len(x) == len(c):
			TP = len(df_positive[(df_positive[models[-1]] >= x[len(c) - 1])])
			FP = len(df_positive[(df_positive[models[-1]] < x[len(c) - 1])])
			TN = len(dfNetagive[(dfNetagive[models[-1]] < x[len(c) - 1])])
			FN = len(dfNetagive[(dfNetagive[models[-1]] >= x[len(c) - 1])])
		else:
			TP = len(df_positive[(df_positive[models[-1]] >= x[len(c) - 1]) & (df_positive[models[-1]] <= x[2*len(c) - 1])])
			FP = len(df_positive[(df_positive[models[-1]] < x[len(c) - 1]) | (df_positive[models[-1]] > x[2*len(c) - 1])])
			TN = len(dfNetagive[(dfNetagive[models[-1]] < x[len(c) - 1]) | (dfNetagive[models[-1]] > x[2*len(c) - 1])])
			FN = len(dfNetagive[(dfNetagive[models[-1]] >= x[len(c) - 1]) & (dfNetagive[models[-1]] <= x[2*len(c) - 1])])
		list_sensitivity.append(TP/(TP + FN))
		list_specificity.append(TN/(TN + FP))
		list_F1.append((2*TP)/((2*TP) + FP + FN))
		list_accuracy.append((TP + TN)/(TP + TN + FP + FN))

	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples, list_sensitivity, list_specificity, list_F1, list_accuracy


if __name__ == "__main__":
	# mainDefault()
	runHTVSP()