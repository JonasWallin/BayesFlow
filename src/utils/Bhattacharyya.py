from __future__ import division
import numpy as np


def bhattacharyya_overlap(mu1, Sigma1, mu2, Sigma2):
    '''
        Computes the bhattacharyya overlap exp(-bhd),
        where bhd is the Bhattacharyya distance between
        two normal distributions.

        mu1     - (d x 1) mean for first distribution
        Sigma1  - (d x d) covariance for first distribution
        mu2     - (d x 1) mean for second distribution
        Sigma2  - (d x d) covariance for second distribution
    '''
    bhd = bhattacharyya_distance(mu1, Sigma1, mu2, Sigma2)
    return np.exp(-bhd).reshape(1, 1)


def bhattacharyya_distance(mu1, Sigma1, mu2, Sigma2):
    '''
        Computes the bhattacharyya distance between
        two normal distributions.

        mu1     - (d x 1) mean for first distribution
        Sigma1  - (d x d) covariance for first distribution
        mu2     - (d x 1) mean for second distribution
        Sigma2  - (d x d) covariance for second distribution
    '''
    mu1 = mu1.reshape(-1, 1)
    mu2 = mu2.reshape(-1, 1)
    mSigma = (Sigma1+Sigma2)/2
    try:
        bhd = (np.dot((mu1-mu2).T, np.linalg.solve(mSigma, mu1-mu2))/8
               + .5*np.log(np.linalg.det(mSigma)/np.sqrt(np.linalg.det(Sigma1)
                                                         * np.linalg.det(Sigma2))))
    except np.linalg.linalg.LinAlgError:
        return np.array(np.nan)
    return bhd

if __name__ == "__main__":
    a = np.ones((1, 3))
    b = np.ones((1, 3))
    Sigma1 = np.eye(3)
    Sigma2 = 2*np.eye(3)
    print(bhattacharyya_overlap(a, Sigma1, b, Sigma2))
    #a *= np.nan
    bhattacharyya_overlap(a, Sigma1, b, Sigma2)
    Sigma1 = np.ones((3, 3))
    print(bhattacharyya_overlap(a, Sigma1, b, Sigma2))
    Sigma2 = 3*np.ones((3, 3))
    print(bhattacharyya_overlap(a, Sigma1, b, Sigma2))
