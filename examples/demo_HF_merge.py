from mpi4py import MPI

import BayesFlow as bf
from BayesFlow.utils import Timer
from BayesFlow import PostProcPar
from BayesFlow.utils import load_and_save as ls

from example_util import fcdata,load_setup_postproc_HF

timer = Timer()

rank = MPI.COMM_WORLD.Get_rank()

try:
    hmres = bf.HMres(prodlog,burnlog,data,metadata)
except:
    if expdir[-1] != '/':
        expdir += '/'
    savedir = expdir + str(run)+'/'
    _,setup_postproc = load_setup_postproc_HF(savedir,setupno)
    postpar = setup_postproc()
    data,metadata = fcdata(dataset,Nevent=Nevent,scale='percentilescale',datadir=datadir)
    burnlog = ls.load_burnlog(savedir)
    prodlog = ls.load_prodlog(savedir)
    hmres = bf.HMres(prodlog,burnlog,data,metadata)
    
hmres.merge(method=postpar.mergemeth,**postpar.mergekws)

timer.timepoint("merging")

if rank == 0:
    hmres.clust_nm.get_pdip()
    hmres.clust_m.get_pdip()
timer.timepoint("computing dip test")

ls.save_object(hmres,savedir)

timer.print_timepoints()