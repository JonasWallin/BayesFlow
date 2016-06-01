# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:15:47 2015

@author: johnsson
"""
from __future__ import division
import numpy as np
try:
    from rpy2.robjects.packages import importr
    from rpy2.rinterface import RRuntimeError
    import rpy2.robjects as robjects
except ImportError as e:
    print("{} --- will not be able to load dataset healthyFlowData".format(e))

from ..utils.dat_util import percentilescale


def load(Nsamp=None, scale=True):
    '''
        Load data from R package 'healthyFlowData'.

        Nsamp	-	number of samples to load. If None, all 20 samples are loaded
        scale	-	should the data be scaled so that the 1st percentile of the pooled data ends up at 0 and the 99th percentile of the pooled data ends up at 1.
    '''
    try:
        robjects.r('library(healthyFlowData)')
    except RRuntimeError:
        base = importr('base')
        base.source("http://www.bioconductor.org/biocLite.R")
        biocinstaller = importr("BiocInstaller")
        biocinstaller.biocLite("BiocUpgrade")
        biocinstaller.biocLite("healthyFlowData")
        robjects.r('library(healthyFlowData)')

    robjects.r('data(hd)')
    J = robjects.r('length(hd.flowSet)')[0]
    if not Nsamp is None:
        J = min(J, Nsamp)
    data = []
    sampnames = []
    donorids = []
    for j in range(J):
        samp = str(j+1)
        sampnames.append('sample'+samp)
        data.append(np.ascontiguousarray(np.array(robjects.r('exprs(hd.flowSet[['+samp+']])'))))
        donorids.append(robjects.r('hd.flowSet@phenoData@data$subject[['+samp+']]')[0])

    marker_lab = [ma for ma in np.array(robjects.r('hd.flowSet@colnames'))]
    metasamp = {'names': sampnames, 'donorid': donorids}
    metadata = {'samp': metasamp, 'marker_lab': marker_lab}
    if scale:
        percentilescale(data)

    return data, metadata

if __name__ == '__main__':
    data, metadata = load()
    print("data = {}".format(data))
    print("metadata = {}".format(metadata))
