import os
import BayesFlow as bf
from BayesFlow.utils import Timer

timer = Timer()

'''
     Load res data
'''

data_kws['marker_lab'] = metadata['marker_lab']
res = bf.HMres.load(savedir, data_kws)#prodlog, burnlog, data, metadata)

res.merge(method=postpar.mergemeth, **postpar.mergekws)

timer.timepoint("merging")

if rank == 0:
    res.get_pdip()
    res.get_pdip(suco=False)
timer.timepoint("computing dip test")

print "res saved to {}".format(savedir)
print "res = {}".format(res)
res.save(savedir)

print "savedir has files {}".format(os.listdir(savedir))
comm.Barrier()

timer.print_timepoints()
