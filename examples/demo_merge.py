import BayesFlow as bf
from BayesFlow.utils import Timer
from BayesFlow import PostProcPar
from BayesFlow.utils import load_and_save as ls

from setup_util import fcdata,get_dir_setup_HF_art

dataset = 'HF'
Nevent = 2000
setupno = '0'
expname = 'test'
run = 11
postpar = PostProcPar(True,'bhat_hier_dip',thr=0.47,lowthr=0.08,dipthr=0.28)

datadir,expdir,setupfile,setup = get_dir_setup_HF_art(dataset,expname,setupno)

savedir = expdir +'run'+ str(run) + '/'

data,metadata = fcdata(dataset,Nevent=Nevent,scale='percentilescale',datadir=datadir)

timer = Timer()

burnlog = ls.load_burnlog(savedir)
prodlog = ls.load_prodlog(savedir)

hmres = bf.HMres(prodlog,burnlog,data,metadata)
hmres.merge(method=postpar.mergemeth,**postpar.mergekws)
timer.timepoint("merging")

hmres.clust_nm.get_pdip()
hmres.clust_m.get_pdip()
timer.timepoint("computing dip test")

ls.save_object(hmres,savedir)

timer.print_timepoints()