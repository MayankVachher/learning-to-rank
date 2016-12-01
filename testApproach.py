from experiment.runSingleExperiment import runSingleExperiment
import numpy as np 
import pickle

def main():

	with open("matrix_factorization_sgd_results", "w") as resFile:

		resFile.write("K=5, steps=50, alpha=0.0002, beta=0.02\n")

		results_ndcg = []
		results_map = []

		for x in range(1,6):
			trainTupl = pickle.load(open('folds/fold'+str(x)+'.train', 'r'))
			testTupl = pickle.load(open('folds/fold'+str(x)+'.test', 'r'))
			
			print "Done loading pickled files ...\n"
			n_users = 943
			n_items = 1682

			trainMat = np.zeros((n_users, n_items))
			
			for tupl in trainTupl:
				uid, mid, r = tupl
				trainMat[uid, mid] = r

			vals = runSingleExperiment(trainMat, testTupl)

			results_ndcg.append(vals[0])
			results_map.append(vals[1])

			resFile.write("Fold "+str(x)+" NDCG: {0:.4f}\n".format(results_ndcg[-1]))
			resFile.write("Fold "+str(x)+" MAP: {0:.4f}\n".format(results_map[-1]))

		print "Average NDCG@5 across Folds: {0:.4f}\n".format(float(sum(results_ndcg))/5)
		resFile.write("Avg. NDCG@4 across Folds: {0:.4f}\n".format(float(sum(results_ndcg))/5))

		print "Average MAP@5 across Folds: {0:.4f}\n".format(float(sum(results_map))/5)
		resFile.write("Avg. MAP@4 across Folds: {0:.4f}\n".format(float(sum(results_map))/5))

if __name__ == '__main__':
	main()