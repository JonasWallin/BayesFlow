# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:10:38 2014

@author: johnsson
"""
from __future__ import division
import numpy as np

def percentilescale(data, q = (1.,99.)):
    alldata = np.vstack(data)
    datq = np.percentile(alldata,q,0)
    for j in range(len(data)):
        for m in range(data[0].shape[1]):
            data[j][:,m] = (data[j][:,m]-datq[0][m])/(datq[1][m]-datq[0][m])
    return datq

            
