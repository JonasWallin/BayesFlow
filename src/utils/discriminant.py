from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def discriminant_projection(data,w1,w2):
    '''
        Projection of a data set onto Fisher's discriminant coordinate between two populations which are defined by weights on the data set.
        
        data    -   (n x d) data set
        w1      -   (n x 1) weights defining first population
        w2      -   (n x 1) weights defining secon population
        
    '''
    mu1,Sigma1 = population_mu_Sigma(data,w1)
    mu2,Sigma2 = population_mu_Sigma(data,w2)
    dc = discriminant_coordinate(mu1,Sigma1,mu2,Sigma2)
    proj = np.dot(data,dc)
    return proj

def population_mu_Sigma(data,weights):
    '''
        Weighted mean and covariance matrix
    '''
    w = weights.reshape((data.shape[0],1)).astype('float')
    w /= np.sum(w)
    wmean = np.sum(data*w,axis=0)
    cdata = data - wmean
    wSigma = np.dot(cdata.transpose(),cdata*w)
    return wmean,wSigma

def discriminant_coordinate(mu1,Sigma1,mu2,Sigma2):
    '''
        Fisher's discriminant coordinate
    '''
    w = np.linalg.solve(Sigma1+Sigma2,mu2-mu1)
    w /= np.linalg.norm(w)
    return w

def plot_discriminant_projection(data,w1,w2,fig = None):
    proj = discriminant_projection(data,w1,w2)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(proj,w1,color='red')
    ax.scatter(proj,w2,color='blue')