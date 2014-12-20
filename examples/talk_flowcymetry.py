# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:24:47 2014

@author: jonaswallin
"""
from __future__ import division
import numpy as np
import BayesFlow as bm
import os
import time
import matplotlib.pyplot as plt
import BayesFlow.plot as bm_plot

if __name__ == '__main__':
	plt.close('all')
	plt.ion()
	sim  = 200
	sim2 = 500
	data = []
	for file_ in os.listdir("../data/flow_dataset/"):
		if file_.endswith(".dat"):
			data.append(np.ascontiguousarray(np.loadtxt("../data/flow_dataset/" + file_)))
	f = bm_plot.histnd(data[0],100,[1, 99],[1,99])
	f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/hist2_singleIndv.pdf", type="pdf",bbox_inches='tight')
	f = bm_plot.histnd(data[1],100,[1, 99],[1,99])
	f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/hist2_singleIndv1.pdf", type="pdf",bbox_inches='tight')
	f = bm_plot.histnd(data[2],100,[1, 99],[1,99])
	f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/hist2_singleIndv2.pdf", type="pdf",bbox_inches='tight')
	f = bm_plot.histnd(data[3],100,[1, 99],[1,99])
	f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/hist2_singleIndv3.pdf", type="pdf",bbox_inches='tight')
	f = bm_plot.histnd(np.vstack(data),100,[1, 99],[1,99])
	f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/hist2_multIndv.pdf", type="pdf",bbox_inches='tight')
	 
