# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:15:47 2015

@author: johnsson
"""
from __future__ import division
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError
import rpy2.robjects as robjects
import BaysFlow.utils.transform as transform

def load(Nsamp = None,scale = True):
    
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
        J = min(J,Nsamp)
    data = []
    sampnames = []
    donorids = []    
    for j in range(J):
        samp = str(j+1)
        sampnames.append('sample '+samp)
        data.append(np.array(robjects.r('exprs(hd.flowSet[['+samp+']])')))
        donorids.append(robjects.r('hd.flowSet@phenoData@data$subject[['+samp+']]')[0])

    marker_lab = [ma for ma in np.array(robjects.r('hd.flowSet@colnames'))]
    metasamp = {'names':sampnames,'donorid': donorids}
    metadata = {'samp':metasamp,'marker_lab':marker_lab}
    if scale:
        transform.percentilescale(data)
        
    return data,metadata
        
#data,metadata = load()
#print "data = {}".format(data)
#print "metadata = {}".format(metadata)