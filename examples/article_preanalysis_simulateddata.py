'''
Created on Aug 10, 2014

@author: jonaswallin
'''

import article_simulatedata
import matplotlib
import BayesFlow.plot as bm_plot
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

if __name__ == "__main__":
	#Y = np.array(article_simulatedata.simulate_data(nCells = 20000, nPersons = 40))
	Y = np.array(article_simulatedata.simulate_data(nCells = 10000, nPersons = 80)[0])
	f = bm_plot.histnd(np.vstack(Y), 100, [0, 100],[0,100])
	
	f_ = bm_plot.histnd(Y[0,:,:],50,[0, 100],[0,100])
	#plt.show()
	
	
	f.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_simulated_obs.eps", type="eps",bbox_inches='tight')
	f.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_simulated_obs.pdf", type="pdf",bbox_inches='tight')
	f_.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_indv_simulated_obs.eps", type="eps",bbox_inches='tight')
	f_.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_indv_simulated_obs.pdf", type="pdf",bbox_inches='tight')
	