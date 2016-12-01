import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle

def main():
	
	names = ['user_id', 'item_id', 'rating', 'timestamp']
	csv_tuples = pd.read_csv("../MovieLens-100k/u.data", sep='\t', names=names)

	user_dict = {}

	for row in csv_tuples.itertuples():
		user, item, rating = row[1]-1, row[2]-1, row[3]
		if user not in user_dict:
			user_dict[user] = [(item, rating)]
		else:
			user_dict[user].append((item, rating))

	kf = KFold(n_splits=5, shuffle=True)

	train_set = {0: [], 1: [], 2: [], 3: [], 4: []}
	test_set = {0: [], 1: [], 2: [], 3: [], 4: []}

	# for train, test in kf.split(user_dict[0]):
	# 	print train
	# 	print test
	# 	print "\n"

	for user in user_dict:
		split_gen = []
		for train, test in kf.split(user_dict[user]):
			split_gen.append((train,test))
		for n_split in range(5):
			fold_split_train, fold_split_test = split_gen[n_split]
			for i, ind in enumerate(fold_split_train):
				u_id = user
				i_id, rating = user_dict[user][ind]
				train_set[n_split].append((u_id, i_id, rating))

			for i, ind in enumerate(fold_split_test):
				u_id = user
				i_id, rating = user_dict[user][ind]
				test_set[n_split].append((u_id, i_id, rating))
		print "Done for User "+str(user)

	print len(train_set[0])
	print len(test_set[0])

	all_data = {"train": train_set, "test": test_set}

	for i in train_set:
		pickle.dump(train_set[i], open("../folds/fold"+str(i+1)+".train", 'w'))

	for i in test_set:
		pickle.dump(test_set[i], open("../folds/fold"+str(i+1)+".test", 'w'))

	# pickle.dump(all_data, open("foldSplit.p", "w"))

if __name__ == '__main__':
	main()