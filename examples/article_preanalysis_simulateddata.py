'''
Created on Aug 10, 2014

@author: jonaswallin
'''

import article_simulatedata
import matplotlib
import BayesFlow.plot as bm_plot
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


folderFigs = "/Users/jonaswallin/Dropbox/articles/FlowCap/figs/"
if __name__ == "__main__":
	nPersons  = 80
	nCells = 15000
	
	rand_pers = npr.choice(nPersons,size=nCells)
	rand_cells = npr.choice(nCells,size=nCells)
	Y = np.array(article_simulatedata.simulate_data(nCells = nCells, nPersons = nPersons)[0])
	
	Y_subsample = np.zeros((nCells,3))
	for i in range(nCells):
		Y_subsample[i,:] = Y[rand_pers[i],rand_cells[i],:]
	
	f = bm_plot.histnd(Y_subsample, 50, [0, 100],[0,100])
	
	f_ = bm_plot.histnd(Y[0,:,:],50,[0, 100],[0,100])
	#plt.show()
	
	#npr.choice(10,size=100)
	f.savefig(folderFigs + "hist2d_simulated_obs.eps", type="eps",bbox_inches='tight')
	f.savefig(folderFigs + "hist2d_simulated_obs.pdf", type="pdf",bbox_inches='tight')
	f_.savefig(folderFigs + "hist2d_indv_simulated_obs.eps", type="eps",bbox_inches='tight')
	f_.savefig(folderFigs + "hist2d_indv_simulated_obs.pdf", type="pdf",bbox_inches='tight')
	