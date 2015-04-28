import BayesFlow.data.healthyFlowData as hf
import os
import numpy as np

datadir = 'healthyFlowData/'
if not os.path.exists(datadir):
	os.makedirs(datadir)
data,metadata = hf.load(scale=False)
for j,dat in enumerate(data):
	np.savetxt(datadir+metadata['samp']['names'][j]+'.dat',data[j])