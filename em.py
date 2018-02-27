import numpy as np
import pylab as plt
import os
import cv2
from scipy.stats import multivariate_normal as mvn

########################################################################################
########################################################################################
#Set Train = 1 for training
Train = 1
K=7

#Set values here
testfolder = "Test Images"
outfolder = "EM Output"
########################################################################################
########################################################################################

def EMalgo(xtrain,K,iters):

	n,d = xtrain.shape            
	mean = xtrain[np.random.choice(n, K, False), :]
	Sigma = [80*np.eye(d)] * K
	for i in range(K):
		Sigma[i]=np.multiply(Sigma[i],np.random.rand(d,d))
		#Sigma[i]=Sigma[i]*Sigma[i].T
	w = [1./K] * K
	z = np.zeros((n, K))

	log_likelihoods = []

	while len(log_likelihoods) < iters:
		for k in range(K):
			#x_mean = np.matrix(xtrain - mean[k])
			#Sinv = np.linalg.pinv(Sigma[k])
			tmp = w[k] * mvn.pdf(xtrain, mean[k], Sigma[k],allow_singular=True)#((2.0 * np.pi) ** (-d / 2.0)) * (1.0 / (np.linalg.det(Sigma[k]) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(x_mean * Sinv, x_mean), axis=1))
			z[:,k]=tmp.reshape((n,))

		log_likelihood = np.sum(np.log(np.sum(z, axis = 1)))

		print '{0} -> {1}'.format(len(log_likelihoods),log_likelihood)
		if log_likelihood>-592596: break

		log_likelihoods.append(log_likelihood)

		z = (z.T / np.sum(z, axis = 1)).T
		 
		N_ks = np.sum(z, axis = 0)
		
		for k in range(K):
			mean[k] = 1. / N_ks[k] * np.sum(z[:, k] * xtrain.T, axis = 1).T
			x_mean = np.matrix(xtrain - mean[k])
			Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mean.T,  z[:, k]), x_mean))
			w[k] = 1. / n * N_ks[k]

		if len(log_likelihoods) < 2 : continue
		if len(log_likelihoods)>10000 or np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001: break

	plt.plot(log_likelihoods)
	plt.title('Log Likelihood vs iteration plot')
	plt.xlabel('Iterations')
	plt.ylabel('log likelihood')
	plt.show()
	np.save('weights',w)
	np.save('sigma',Sigma)
	np.save('mean',mean)

xtrain = np.load('xtrain.npy')

if(Train == 1):
	EMalgo(xtrain,K,10000)

w=np.load('weights.npy')
Sigma=np.load('sigma.npy')
mean=np.load('mean.npy')

for filename in os.listdir(testfolder):

	img = cv2.imread(os.path.join(testfolder,filename))
	cv2.imshow('Test', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	basename = os.path.basename(filename)
	print basename
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	nr, nc, d = img.shape
	n=nr*nc
	xtest=np.reshape(img,(n,d))
	likelihoods=np.zeros((K,n))
	log_likelihood=np.zeros(n)
	for k in range(K):
		#x_mean = np.matrix(xtrain - mean[k])
		#Sinv = np.linalg.pinv(Sigma[k])
		likelihoods[k] = w[k] * mvn.pdf(xtest, mean[k], Sigma[k],allow_singular=True)#((2.0 * np.pi) ** (-d / 2.0)) * (1.0 / (np.linalg.det(Sigma[k]) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(x_mean * Sinv, x_mean), axis=1))
		log_likelihood = likelihoods.sum(0)

	log_likelihood = np.reshape(log_likelihood, (nr, nc))

	log_likelihood[log_likelihood > np.max(log_likelihood) / 1.18] = 255 
	output_img = np.zeros(img.shape)
	output_img[:, :, 2] = log_likelihood
	output_img = np.array(output_img)
	output_img = cv2.resize(output_img, (400, 300))

	cv2.imshow('RED',output_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite(os.path.join(outfolder, basename[:-4]+'RED'+'.png'), output_img)