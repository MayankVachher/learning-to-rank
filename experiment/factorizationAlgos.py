import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.decomposition import NMF
# import pymf

def softimpute(Q):
	Q[Q == 0] = np.nan
	return SoftImpute(max_iters=20, max_rank=2, min_value=1, max_value=5).complete(Q)

def basicMF(Q):
	print "Started basic MF\n"
	model = NMF(init='random', random_state=0, max_iter=20)
	res = model.fit(Q)
	return res

def basicMF2(Q):
	print "Started basic MF2\n"
	R = Q
	N = len(R)
	M = len(R[0])
	K = 5
	p = np.random.rand(N,K)
	q = np.random.rand(M,K)
	nP, nQ = matrix_factorization(R, p, q, K)
	nR = np.dot(nP, nQ.T)
	return nR

def matrix_factorization(R, P, Q, K=5, steps=50, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
    	print step
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T
