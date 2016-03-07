'''
Created on Aug 10, 2014

@author: jonaswallin
'''

import article_simulatedata_old
import matplotlib
import BayesFlow.plot as bm_plot
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import os
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
#pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
#matplotlib.rcParams.update(pgf_with_rc_fonts)

folderFigs = "/Users/jonaswallin/Dropbox/articles/FlowCap/figs/tmp/"
if __name__ == "__main__":
	nPersons  = 40
	nCells = 10000
	
	rand_pers = npr.choice(nPersons,size=nCells)
	rand_cells = npr.choice(nCells,size=nCells)
	#Y = np.array(article_simulatedata.simulate_data_v1(nCells = nCells, nPersons = nPersons)[0])
	Y = np.array(article_simulatedata_old.simulate_data(nCells = nCells, nPersons = nPersons)[0])
	Y_subsample = np.zeros((nCells,3))
	for i in range(nCells):
		Y_subsample[i,:] = Y[rand_pers[i],rand_cells[i],:]
	
	f = bm_plot.histnd(Y_subsample, 50, [0, 100],[0,100], lims = np.array([[-3.8, 3.2], [-3.6, 3.9], [-1.8, 2.9]]))
	
	f_ = bm_plot.histnd(Y[0,:,:],50,[0, 100],[0,100], lims = np.array([[-3.1, 2.5], [-3.4, 4.1], [-1.8, 2.3]]))
	#plt.show()
	
	#npr.choice(10,size=100)
	print folderFigs + "hist2d_simulated_obs.eps"
	print os.getcwd()
	f.savefig(folderFigs + "hist2d_simulated_obs.eps", type="eps",bbox_inches='tight')
	f.savefig(folderFigs + "hist2d_simulated_obs.pdf", type="pdf",bbox_inches='tight')
	f_.savefig(folderFigs + "hist2d_indv_simulated_obs.eps", type="eps",bbox_inches='tight')
	f_.savefig(folderFigs + "hist2d_indv_simulated_obs.pdf", type="pdf",bbox_inches='tight')
	