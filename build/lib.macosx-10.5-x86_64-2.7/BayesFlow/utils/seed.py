import time
import re
from mpi4py import MPI

def get_seed(expname):
	"""
		Get a random seed based on expname and current time.

		expname 	-	string. Only numbers inside affect result
	"""
	if MPI.COMM_WORLD.Get_rank() == 0:
		s = expname+str(time.time())
		return int(''.join(re.findall(r'\d+',s)))%4294967295
	return None