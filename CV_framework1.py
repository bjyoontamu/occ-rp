import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds, fsolve, brentq
from scipy.stats import mvn, gaussian_kde

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
	
	kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
	index_dataset = 0
	for test_index, train_index in kf.split(df, Y):
		if len(regressor) == 0:
			df_train, df_test = pd.concat([df.iloc[train_index], df_remaining]).copy(), df.iloc[test_index].copy()
		else:
			df_train, df_test = evaluateSamples(regressor, pd.concat([df.iloc[train_index], df_remaining]).copy(), df.iloc[test_index].copy(), features, feature_causality)
		evaluateOCC(lambdaLast, models, index_dataset, df_train, df_test, algorithms, "EM", prinprintOutput, isQuick, screeningApproach)
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

def evaluateOCC(lambdaLast, models, index_dataset, df_training, df_test, algorithms, densityEstimation = "MLE", prinprintOutput = False, isQuick = False, screeningApproach = "hard"):
	defaultCost = np.array([7, 4, 32949, 2, 33991, 7890])

	X0 = len(df_test)
	if len(lambdaLast) == 1:
		GT = len(df_test[df_test[models[-1]] >= lambdaLast[0]])
	else:
		GT = len(df_test[(df_test[models[-1]] >= np.min(lambdaLast)) & (df_test[models[-1]] <= np.max(lambdaLast))])
	# return
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
	# stageSetups = [(0, 1, 2, 3)]
	stageSetups = [(0, 1, 2, 3, 4, 5)]
	data = {}
	data['costPerStage'] = defaultCost
	data['numberOfSamples'] = X0
	data['GT'] = GT
	i = 0
	for stageSetup in stageSetups: 
		# if len(stageSetup) != 4:
		# 	continue
		stageSetup = np.array(stageSetup)

		f_s = estimateScoreDensity(densityEstimation, df_training[models[stageSetup]])
		
		reorderedC = np.array(defaultCost)
		reorderedC = reorderedC[stageSetup]

		listOutput = list() 
		prevOperator = np.empty(len(lambdaLast)* (len(stageSetup)-1))
		prevOperator[:] = np.nan

		evaluationRange = np.concatenate((np.arange(0, 74828.29115*X0, (74828.29115*X0)/20), np.array([74828.29115*X0])))
		# evaluationRange = np.concatenate((np.arange(0, 74828.29115*X0, (74828.29115*X0)/5), np.array([74828.29115*X0])))
		# evaluationRange = np.concatenate((np.arange((74828.29115*X0)/5, 74828.29115*X0, (74828.29115*X0)/5), np.array([74828.29115*X0])))
		for CTotali in evaluationRange:
			dicOutput, prevOperator = runAlgorithms(df_test, models[stageSetup], algorithms, prevOperator, f_s, X0, reorderedC, CTotali, lambdaLast, GT, True, stageSetup, prinprintOutput, isQuick, screeningApproach)
			dicOutput['x'] = CTotali
			listOutput.append(dicOutput)
			outAlgorithms = dicOutput['algorithms']
			standardMethod = 0
			for i in range(len(outAlgorithms)):
				if 'proposed' in outAlgorithms[i]:
					standardMethod = i
					break
			if dicOutput[outAlgorithms[standardMethod] + '_samples'][-1] >= GT:
				break

		data[''.join(map(str, stageSetup))] = listOutput

	if len(lambdaLast) == 1:
		resultPath = './results/data' + "".join(map(str, np.int0(defaultCost))) + '/'
	else:
		resultPath = './results/data' + "".join(map(str, np.int0(defaultCost))) + 'r/'
	if not os.path.isdir(resultPath):
		os.mkdir(resultPath) 
	with open(resultPath +  'CV' + str(index_dataset) + '.pickle', 'wb') as handle:
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

def runAlgorithms(df, models, algorithms, prevOperator, f_s = dict({"method": "invalid"}),
			X0 = 10**5, c = [1,10,100,1000], C_total = (100*2*(10**5)), lambdaLast = 0, GT = 0, sampleEvaluation=False, stageSetup = 0, printOutput = False, isQuick = False, screeningApproach = "hard"):

	assert(f_s["method"] != "invalid")

	if f_s["method"] == "EM" or f_s["method"] == "MLE":
		pi = f_s['pi']
		mu = f_s['mu']
		sigma = f_s['sigma']
	
	numberOfStages = len(c)
	violationTolerance  = 0.01

	dicOutput = {}
	dicOutput['algorithms'] = algorithms
	if (X0*c[0] >= C_total):
		print("!!!!Warning!!!! invalid parameter!")
		for i in range(len(algorithms)):
			dicOutput[algorithms[i] + '_samples'] = (0, 0)
		return dicOutput, prevOperator

	print("Number of true candidates: {:.0f}".format(GT))
	for i in range(len(algorithms)):
		lambdas = np.concatenate((np.asarray([0]*len(lambdaLast)*(numberOfStages-1)), lambdaLast))
		lambdas = np.concatenate((lambdas[0: : 2], lambdas[1: : 2])) # [lower bounds, upper bounds]

		if algorithms[i] == 'proposedLambda':
			operator_temp, dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = proposedApproach(f_s, lambdaLast, C_total, c, X0, prevOperator, isQuick)
			prevOperator = operator_temp
			if len(lambdaLast) == 1:
				lambdas = np.concatenate((operator_temp, lambdaLast))
			else:
				lambdas = np.concatenate((operator_temp[: len(c) - 1], [lambdaLast[0]], operator_temp[len(c) - 1:], [lambdaLast[1]]))
			_, samples = validateOperatorAnalytically(lambdas, f_s, C_total, c, X0, violationTolerance, algorithm=algorithms[i], printOutput = printOutput)
		elif algorithms[i] == 'baseline':
			# lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'] = baseApproach2(pi, mu, sigma, lambdas[-1], C_total, c, X0, violationTolerance)
			lambdas[:-1], dicOutput[algorithms[i] + '_cStar'], dicOutput[algorithms[i] + '_time'], samples = baseApproachMine(df.copy(), models, dicOutput['proposedLambda_samples'][-1], stageSetup, pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance)
		if sampleEvaluation:
			dicOutput[algorithms[i] + '_cost'], dicOutput[algorithms[i] + '_samples'], dicOutput[algorithms[i] + 'sensitivity'], dicOutput[algorithms[i] + 'specificity'], dicOutput[algorithms[i] + 'F1'] , dicOutput[algorithms[i] + 'accuracy'] = validateOperatorBySamples(df.copy(), models, lambdas, samples, f_s, C_total, c, X0, violationTolerance, stageSetup, algorithm=algorithms[i], printOutput = printOutput, screeningApproach = screeningApproach)
		else:
			dicOutput[algorithms[i] + '_cost'], dicOutput[algorithms[i] + '_samples'] = validateOperatorAnalytically(lambdas, pi, mu, sigma, C_total, c, X0, violationTolerance, algorithm=algorithms[i], printOutput = printOutput)
		dicOutput[algorithms[i] + '_lambdas'] = lambdas
	return dicOutput, prevOperator

def initialFunction(x, pi, mu, sigma, C_total, c, X0, target):
	lowerBound = [-np.Inf]*(len(mu[0])-1) + [x[0]]
	AUC = (pi[0]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[0]), mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, [np.Inf]*len(mu[1]), mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
	return (target- (AUC*X0))

def proposedApproach(f_s, lambdaLast, C_total, c, X0, prevOperator, isQuick = False):
	p = problem(f_s, lambdaLast, C_total, c, X0)
	nlc = NonlinearConstraint(p.constraintFunction, -np.inf, C_total, hess = lambda x, v: np.zeros((len(lambdaLast)*(len(c)-1), len(lambdaLast)*(len(c)-1))))

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

	C_used = 0
	iterationCount = 0
	bestOperators = np.zeros(len(lambdaLast)*(len(c)-1))
	bestFunction = 0
	bestTime = 0
	bestBudget = 0
	while np.abs(C_total-C_used) > (C_total*0.1):
		if iterationCount > 0:
			print("It seems it is not good, lets try again ({})".format(iterationCount))
		start = time.time()
		if isQuick:
			if np.isnan(prevOperator).any():
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc))
			else:
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), x0 = prevOperator)
		else:
			if np.isnan(prevOperator).any():
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-20), mutation=[0.05, 1])
			else:
				result = differential_evolution(p.objectiveFunction, bounds, constraints=(nlc), tol=10**(-20), mutation=[0.05, 1], x0 = prevOperator)

		end = time.time()
		if len(lambdaLast) == 1:
			C_used =  p.findBudgetFromLambdas(np.append(result.x, lambdaLast))
		else:
			param = np.concatenate((result.x[: len(c) - 1], [lambdaLast[0]], result.x[len(c) - 1:], [lambdaLast[1]]))
			C_used =  p.findBudgetFromLambdas(param)
		if np.abs(C_total-C_used) < np.abs(C_total-bestBudget):
			bestBudget = C_used
			bestOperators = result.x
			bestTime = (end - start)
			bestFunction = result.fun
		iterationCount += 1
		if iterationCount > 100:
			print("OMG..")
			sendEmail("OMG..")
			break
	return bestOperators, bestFunction, bestTime

class problem:
	def __init__(self, f_s, lambdaLast, C_total, c, X0):
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
		self.C_total = C_total
		self.c = c
		self.X0 = X0
		self.upperBound = [np.Inf]*len(self.c)

	def objectiveFunction(self, integrationBound):
		if self.method == "EM" or self.method == "MLE":
			if len(self.lambdaLast) == 1:
				op = (self.pi[0]*mvn.mvnun(np.append(integrationBound, self.lambdaLast), self.upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(np.append(integrationBound, self.lambdaLast), self.upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
			else:
				op = (self.pi[0]*mvn.mvnun(np.append(integrationBound[: len(self.mu[0])-1], self.lambdaLast.min()), np.append(integrationBound[len(self.mu[0])-1:], self.lambdaLast.max()), self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(np.append(integrationBound[: len(self.mu[0])-1], self.lambdaLast.min()), np.append(integrationBound[len(self.mu[0])-1:], self.lambdaLast.max()), self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
		else:
			op = self.f_s.integrate_box(np.append(integrationBound[: len(self.c)-1], self.lambdaLast.min()), np.append(integrationBound[len(self.c)-1:], self.lambdaLast.max()), maxpts=len(self.c)*10000)
		return (1 - op)

	def constraintFunction(self, integrationBound):
		LB = [-np.Inf]*len(self.c)
		UB = [np.Inf]*len(self.c)

		cumulativeCost = self.c[0]*self.X0
		if self.method == "EM" or self.method == "MLE":
			if len(self.lambdaLast) == 1:
				for i in range(len(self.c)-1):
					LB[i] = integrationBound[i]
					AUC = (self.pi[0]*mvn.mvnun(LB, self.upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(LB, self.upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
					cumulativeCost += (self.c[i+1]*self.X0*AUC)
			else:
				for i in range(len(self.c)-1):
					LB[i] = integrationBound[i]
					UB[i] = integrationBound[len(self.c)-1 + i]
					AUC = (self.pi[0]*mvn.mvnun(LB, UB, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(LB, UB, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
					cumulativeCost += (self.c[i+1]*self.X0*AUC)
		else:
			for i in range(len(self.c)-1):
				LB[i] = integrationBound[i]
				UB[i] = integrationBound[len(self.c)-1 + i]
				AUC = self.f_s.integrate_box(LB, UB, maxpts=len(self.c)*10000)
				cumulativeCost += (self.c[i+1]*self.X0*AUC)
		return np.array(cumulativeCost)

	def findBudgetFromLambdas(self, x):
		lowerBound = [-np.Inf]*len(self.c)
		upperBound = [np.Inf]*len(self.c)
		computationalCostVector = [0]*(len(self.c))
		numberOfPassedSamples = [self.X0] + [0]*(len(self.c))

		if len(x) > len(self.c):
			for i in range(len(self.c)):
				computationalCostVector[i] = (self.c[i]*numberOfPassedSamples[i])
				lowerBound[i] = x[i]
				upperBound[i] = x[i + len(self.c)]
				if self.method == "EM" or self.method == "MLE":
					AUC = (self.pi[0]*mvn.mvnun(lowerBound, upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(lowerBound, upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
				else:
					AUC = self.f_s.integrate_box(lowerBound, upperBound, maxpts=len(self.c)*10000)
				numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*AUC
		else:
			for i in range(len(x)):
				computationalCostVector[i] = (self.c[i]*numberOfPassedSamples[i])
				lowerBound[i] = x[i]
				AUC = (self.pi[0]*mvn.mvnun(lowerBound, self.upperBound, self.mu[0], self.sigma[0], maxpts=len(self.mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (self.pi[1]*mvn.mvnun(lowerBound, self.upperBound, self.mu[1], self.sigma[1], maxpts=len(self.mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])
				numberOfPassedSamples[i+1] = numberOfPassedSamples[0]*AUC

		return np.sum(computationalCostVector)

def integrateCost(pi, mu, sigma, lowerBound, upperBound):
	return (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(mu[0])*10000, abseps = 1e-10, releps = 1e-10)[0]) + (pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0])

def baseApproachMine(dfOriginal, models, targetSamples, stageSetup, pi, mu, sigma, lambdaLast, C_total, c, X0, violationTolerance):
	# # version 2
	# start = time.time()
	# decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, pi, mu, sigma, lambdaLast, c, C_total, X0))
	# # decayRatio = 0.3 # pass 30 % of samples
	# lambdas = [-np.Inf]*(len(mu[0])-1)
	# for i in range(len(lambdas)):
	# 	initialPoint = 0
	# 	lambdas[i] = fsolve(baselineFindLambdas, initialPoint, args=(i, lambdas, pi, mu, sigma, lambdaLast, c, X0, decayRatio), maxfev=1000000)[0]
	# end = time.time()
	# f_star = 1 - integrateCost(pi, mu, sigma, lambdas + [lambdaLast], [np.Inf]*len(mu[0]))
	# return lambdas, decayRatio, (end - start)
	
	# version 3
	start = time.time()
	decayRatio = brentq(findDecayRatio, 0, 1, args=(targetSamples, dfOriginal, models, pi, mu, sigma, lambdaLast, c, C_total, X0))
	lambdas = [-np.Inf]*(len(mu[0])-1)
	reorderedTable = dfOriginal[models]
	numberOfPassedSamples = [X0] + [0]*(len(models))
	totalComputationalCost = 0
	for i in range(len(models)):
		totalComputationalCost += (numberOfPassedSamples[i]*c[i])
		if i < (len(models)-1):
			oneAddreorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)) + 1, models[i])
			reorderedTable = reorderedTable.nlargest(int(decayRatio*len(reorderedTable)), models[i])
			lambdas[i] = (np.min(reorderedTable[models[i]]) + np.min(oneAddreorderedTable[models[i]]))/2
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[models[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)
	end = time.time()
	if (C_total*1.0001 < totalComputationalCost) or (np.abs(C_total - totalComputationalCost) > (C_total*violationTolerance)):
		sendEmail("Baseline works incorrectly")

	return lambdas, decayRatio, (end - start), numberOfPassedSamples

def findDecayRatio(r, targetSamples, df, models, pi, mu, sigma, lambdaLast, c, C_total, X0):
	reorderedTable = df[models]
	computationalCostVector = [0]*(len(models))
	numberOfPassedSamples = [X0] + [0]*(len(models))
	for i in range(len(models)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		
		if i < (len(models)-1):
			reorderedTable = reorderedTable.nlargest(int(r*len(reorderedTable)), models[i])
			numberOfPassedSamples[i+1] = len(reorderedTable)
		else:
			reorderedTable = reorderedTable[reorderedTable[models[i]] > lambdaLast]
			numberOfPassedSamples[i+1] = len(reorderedTable)

	return (C_total - np.sum(computationalCostVector))

def baselineFindLambdas(x, i, LB, pi, mu, sigma, lambdaLast, c, X0, decayRatio):
	LB[i] = x[0]
	# Xi = mvn.mvnun(LB + [-np.Inf], [np.Inf]*len(mu), pi, mu, sigma, maxpts=len(mu)*10000, abseps = 1e-10, releps = 1e-10)[0]
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

def validateOperatorAnalytically(x, f_s, C_total, c, X0, violationTolerance, algorithm = "unknown", printOutput = True):
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
	computationalCostVector = [0]*(len(c))
	numberOfPassedSamples = [X0] + [0]*(len(c))

	for i in range(len(c)):
		computationalCostVector[i] = (c[i]*numberOfPassedSamples[i])
		lowerBound[i] = x[i]
		if len(x) != len(c):
			upperBound[i] = x[i + len(c)]

		if method == "EM" or method == "MLE":
			AUC = (pi[0]*mvn.mvnun(lowerBound, upperBound, mu[0], sigma[0], maxpts=len(c)*10000, abseps = 1e-10, releps = 1e-10)[0]) + ((pi[1]*mvn.mvnun(lowerBound, upperBound, mu[1], sigma[1], maxpts=len(mu[1])*10000, abseps = 1e-10, releps = 1e-10)[0]))
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
	if C_total < sum(computationalCostVector):
		if np.abs(C_total - sum(computationalCostVector)) > (C_total*violationTolerance):
			print("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
			sendEmail("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	print("\tSamples passed:{:.2f}, Maximum cost allowed: {}, Total computational cost: {}".format(numberOfPassedSamples[-1], C_total, sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples

def validateOperatorBySamples(df, models, x, samples, f_s, C_total, c, X0, violationTolerance, stageSetup, algorithm = "unknown", printOutput = True, screeningApproach = "hard"):
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

	if C_total < sum(computationalCostVector):
		if np.abs(C_total - sum(computationalCostVector)) > (C_total*violationTolerance):
			sendEmail("!!!!Warning!!!! - maximum cost allowed: {}, but total computational cost: {}".format(C_total, sum(computationalCostVector)))
	print("\tSamples passed:{:.2f}, Maximum cost allowed: {}, Total computational cost: {}".format(numberOfPassedSamples[-1], C_total, sum(computationalCostVector)))
	
	print("\tSamples passed:{:.2f}, Total computational cost: {}".format(numberOfPassedSamples[-1], sum(computationalCostVector)))
	return computationalCostVector, numberOfPassedSamples, list_sensitivity, list_specificity, list_F1, list_accuracy

if __name__ == "__main__":
	runHTVSP()
