'''
Created on Nov 30, 2015

@author: jonaswallin
'''
import numpy as np



import BayesFlow.plot as bm_plot
folderFigs = "/Users/jonaswallin/Dropbox/articles/FlowCap/figs/tmp/"
#simulation_result=np.load('/Users/jonaswallin/repos/BaysFlow/script/simulation_result.npy').item()
#Y0 = simulation_result['Y_0']


res=np.load('/Users/jonaswallin/repos/BaysFlow/examples/article1/sim_data_v1.npy')
Y = np.array(res[0])
f = bm_plot.histnd(Y, 50, [0, 100],[0,100], lims = np.array([[-3.8, 3.2], [-3.6, 3.9], [-1.8, 2.9]]))

f_ = bm_plot.histnd(Y[0,:,:],50,[0, 100],[0,100], lims = np.array([[-3.1, 2.5], [-3.4, 4.1], [-1.8, 2.3]]))
#plt.show()

#npr.choice(10,size=100)
print folderFigs + "hist2d_simulated_obs.eps"
#print os.getcwd()
f.savefig(folderFigs + "hist2d_simulated_obs.eps", type="eps",bbox_inches='tight')
f.savefig(folderFigs + "hist2d_simulated_obs.pdf", type="pdf",bbox_inches='tight')
f_.savefig(folderFigs + "hist2d_indv_simulated_obs.eps", type="eps",bbox_inches='tight')
f_.savefig(folderFigs + "hist2d_indv_simulated_obs.pdf", type="pdf",bbox_inches='tight')