.PHONY: default, article

MPIEXEC=mpiexec -n 4
PYTHON=python

article:
	echo 100 | ${MPIEXEC} ${PYTHON} article_simulated_estimate_mpi.py

default:
	echo 100 | ${MPIEXEC} ${PYTHON} flowcymetry_hGMM_mpi.py
	
example:
	${MPIEXEC} ${PYTHON} example_hGMM_simulated_MPI.py

HF:
	echo 100 | ${MPIEXEC} ${PYTHON} HF_MCMC.py