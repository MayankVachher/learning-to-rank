import numpy as np 
import pandas as pd
from ranking_metrics import ndcg_at_k, mean_average_precision, average_precision
from factorizationAlgos import *
import pickle 
from sklearn.metrics import mean_absolute_error

n_users = 943
n_items = 1682

def runSingleExperiment(trainMat, testTupl):
	item_topic_dist = pickle.load(open('item_topic_dist_100.p','rb'))
	trainMat = np.dot(trainMat, item_topic_dist.T)
	
	testMat = np.zeros((n_users, n_items))
	for tupl in testTupl:
				uid, mid, r = tupl
				testMat[uid, mid] = r

	idealList = getRankedDictForUsers(testTupl)
	print "Done Retrieving Ideal List ...\n"

	ideal = []
	userBias = {}

	for x in idealList:
		currentUser = idealList[x]
		ideal.append(ndcg_at_k([temp[1] for temp in currentUser], len(currentUser)))

		train_ = trainMat[x,]
		fin = train_[train_ != 0]

		uB = sum(fin)/(len(fin))
		userBias[x] = uB

	print "Average NDCG for Ideal: "+str(float(sum(ideal))/len(idealList))

	retrievedMat = retrieve(trainMat)
	print "Done Retrieving Matrix ...\n"

	retrievedMat = np.dot(retrievedMat, item_topic_dist)
	
	retrievedList = getRankedDictForUsers(testTupl, True, retrievedMat)
	print "Done Retrieving Retrieval List ...\n"

	retrieved_ndcg4 = []
	retrieved_map4 = []
	for x in retrievedList:
		currentUser = retrievedList[x]
		retrieved_ndcg4.append(ndcg_at_k([temp[1] for temp in currentUser], 4))

		prec_pre = [0,0,0,0]
		for j in range(4):
			if currentUser[j][1] >= userBias[x]:
				prec_pre[j] = 1

		retrieved_map4.append(mean_average_precision([prec_pre]))

	avg_ndcgat4 = float(sum(retrieved_ndcg4))/len(retrievedList)
	avg_map4 = float(sum(retrieved_map4))/len(retrievedList)
	mae = calcmae(testMat,retrievedMat)
	nmae = (mae/4.0)

	print "Average NDCG@4 for Retrieved: "+str(avg_ndcgat4)
	print "Average MAP@4 for Retrieved: "+str(avg_map4)
	print "NMAE for Retrieved: " +str(nmae)

	return avg_ndcgat4, avg_map4

def retrieve(trainMat):
	return softimpute(trainMat)
	# return basicMF(trainMat)
	# return basicMF2(trainMat)

def getRankedDictForUsers(testTupl, haveRet=False, retMatrix=None):
	
	user_dict = {}

	for tupl in testTupl:
		uid = tupl[0]
		mid = tupl[1]
		r = tupl[2]
		r2 = None

		if haveRet != False:
			r2 = retMatrix[uid, mid]

		if uid not in user_dict:
			user_dict[uid] = [[mid, r, r2]]
		else:
			user_dict[uid].append([mid, r, r2])

	final_dict = {}

	for user in user_dict:
		if haveRet == False:
			user_dict[user] = sorted(user_dict[user], key=lambda x: x[1], reverse=True)
		else:
			user_dict[user] = sorted(user_dict[user], key=lambda x: x[2], reverse=True)

	return user_dict

def calcmae(testmx, predictedmx):
    predictedmx = predictedmx[testmx.nonzero()].flatten()
    testmx = testmx[testmx.nonzero()].flatten()

    return mean_absolute_error(testmx, predictedmx)