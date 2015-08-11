from __future__ import division
import numpy as np

def bhattacharyya_dist(mu1,Sigma1,mu2,Sigma2):
    '''
        Computes exp(-bhd), where bhd is the Bhattacharyya distance between two normal distributions.
        mu1 and mu2 should be column vectors
    '''
    mSigma = (Sigma1+Sigma2)/2
    bhd = np.dot(mu1-mu2,np.linalg.solve(mSigma,mu1-mu2))/8 + .5*np.log(np.linalg.det(mSigma)/np.sqrt(np.linalg.det(Sigma1)*np.linalg.det(Sigma2)))
    if np.isnan(bhd):
        return 0
    return np.exp(-bhd)