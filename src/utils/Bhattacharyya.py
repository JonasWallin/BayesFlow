from __future__ import division
import numpy as np

def bhattacharyya_dist(mu1,Sigma1,mu2,Sigma2):
    '''
        Computes exp(-bhd), where bhd is the Bhattacharyya distance between two normal distributions.
        mu1 and mu2 should be column vectors
    '''
    mu1 = mu1.reshape(-1,1)
    mu2 = mu2.reshape(-1,1)
    mSigma = (Sigma1+Sigma2)/2
    bhd = np.dot((mu1-mu2).T,np.linalg.solve(mSigma,mu1-mu2))/8 + .5*np.log(np.linalg.det(mSigma)/np.sqrt(np.linalg.det(Sigma1)*np.linalg.det(Sigma2)))   
    #if np.isnan(bhd):
    #    return 0
    return np.exp(-bhd)
    
    
if __name__ == "__main__":
    a = np.ones((1,3))
    b = np.ones((1,3))
    Sigma1 = np.eye(3)
    Sigma2 = 2*np.eye(3)
    bhattacharyya_dist(a,Sigma1,b,Sigma2)
    a *= np.nan
    bhattacharyya_dist(a,Sigma1,b,Sigma2)