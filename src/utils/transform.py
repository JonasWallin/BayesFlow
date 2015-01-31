# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:10:38 2014

@author: johnsson
"""
from __future__ import division
import numpy as np

def percentilescale(data, q = (1.,99.)):
    '''
        Scales the data sets in data so that given quantiles of the pooled data ends up at 0 and 1 respectively.

        data	-	list of data sets
	q	-	percentiles. q[0] is the percentile will be scaled to 0, q[1] is the percentile that will be scaled to 1 (in the pooled data).
    '''
    alldata = np.vstack(data)
    datq = np.percentile(alldata,q,0)
    for j in range(len(data)):
        for m in range(data[0].shape[1]):
            data[j][:,m] = (data[j][:,m]-datq[0][m])/(datq[1][m]-datq[0][m])
    return datq

            
