import BayesFlow as bf
from BayesFlow.utils import Timer
from BayesFlow.utils import load_and_save as ls

from example_util import HF,load_setup_postproc_HF

timer = Timer()

try:
    hmres = bf.HMres(prodlog,burnlog,data,metadata)
except:
    if expdir[-1] != '/':
        expdir += '/'
    savedir = expdir + str(run)+'/'
    _,setup_postproc = load_setup_postproc_HF(savedir,setupno)
    postpar = setup_postproc()
    data,metadata = HF(dataset,Nevent=Nevent,scale='percentilescale')
    burnlog = ls.load_burnlog(savedir)
    prodlog = ls.load_prodlog(savedir)
    hmres = bf.HMres(prodlog,burnlog,data,metadata)
    
hmres.merge(method=postpar.mergemeth,**postpar.mergekws)

timer.timepoint("merging")

if rank == 0:
    hmres.clust_nm.get_pdip()
    hmres.clust_m.get_pdip()
timer.timepoint("computing dip test")

print "hmres saved to {}".format(savedir)
print "hmres = {}".format(hmres)
ls.save_object(hmres,savedir)
import os
print "savedir has files {}".format(os.listdir(savedir))
comm.Barrier()

timer.print_timepoints()