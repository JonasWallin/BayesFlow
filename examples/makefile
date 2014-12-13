.PHONY: default, article

MPIEXEC=mpiexec -n 10
PYTHON=python

article:
	echo 100 | ${MPIEXEC} ${PYTHON} article_simulated_estimate_mpi.py

default:
	echo 100 | ${MPIEXEC} ${PYTHON} flowcymetry_hGMM_mpi.py
	
example:
	${MPIEXEC} ${PYTHON} example_hGMM_simulated_MPI.py
