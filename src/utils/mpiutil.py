from mpi4py import MPI
import numpy as np

def collect_int(i,comm = MPI.COMM_WORLD,root=0):
    rank = comm.Get_rank()
    if rank == root:
        i_all = np.empty(comm.Get_size(),dtype = 'i')
    else:
        i_all = np.array([])
    comm.Gather(sendbuf=[np.array(i,dtype='i'), MPI.INT], recvbuf=[i_all, MPI.INT], root=root)
    return i_all

def bcast_int(i,comm = MPI.COMM_WORLD,root=0,):
    rank = comm.Get_rank()
    if rank == root:
        ib = np.array([i],dtype='i')
    else:
        ib = np.array([-1],dtype='i')
    comm.Bcast([ib,MPI.INT],root=root)
    return ib[0]

def bcast_string(s,comm = MPI.COMM_WORLD,root=0):
    rank = comm.Get_rank()
    if rank == root:
        s_list = [np.array(s) for i in range(comm.Get_size())]
    else:
        s_list = None
    return str(comm.scatter(s_list, root = root))

# def bcast_strings(s_list,root=0):
#     if 0:
#         if rank == root:
#             s = ''.join([s_ for s_ in s_list])
#             lens = [len(s_) for s_ in s_list]
#             split_ind = np.hstack([0,np.cumsum(lens,dtype='i')])
#             print "split_ind.shape = {}".format(split_ind.shape)
#         else:
#             s = None
#             split_ind = None
#         s = bcast_string(s,root=root)
#         split_ind = bcast_array_1d(split_ind,'i',MPI.INT,root=root)
#         s_list = [s[split_ind[i]:split_ind[i+1]] for i in range(len(split_ind)-1)]
#     comm.bcast(s_list,root=root)
#     return s_list

def bcast_array_1d(array,nptype,mpitype,comm = MPI.COMM_WORLD,root=0):
    rank = comm.Get_rank()
    if rank == root:
        n = array.shape[0]
    else:
        n = None
    n = bcast_int(n,root=root,comm=comm)
    #print "n = {}".format(n)

    if rank != root:
        array = np.empty(n,nptype)
        print "array = {}".format(array)
    comm.Bcast([array,mpitype],root=root)
    return array

def collect_arrays(data,nbr_col,nptype,mpitype,comm = MPI.COMM_WORLD,root=0):

    nbr_row = collect_data_1d([dat.shape[0] for dat in data],'i',MPI.INT,comm,root=root)
    
    if len(data) > 0:
        data_stack = np.vstack([np.array(dat,dtype=nptype).reshape((-1,nbr_col)) for dat in data])
    else:
        data_stack = np.empty((0,nbr_col),dtype=nptype)

    data_all_stack = collect_array_(data_stack,nbr_col,nptype,mpitype,comm,root=root)
    
    if not data_all_stack is None:
        if len(nbr_row) > 0:
            data_all = np.split(data_all_stack, np.cumsum(nbr_row[0:-1]))
        else:
            data_all = data_all_stack
    else:
        data_all = None
        
    return data_all

def collect_data(data,nbr_col,nptype,mpitype,comm = MPI.COMM_WORLD,root=0):

    if len(data) > 0:
        data_array = np.array(data,dtype=nptype).reshape((-1,nbr_col))
    else:
        data_array = np.empty((0,nbr_col),dtype=nptype)
        
    data_all = collect_array_(data_array,nbr_col,nptype,mpitype,comm,root=root)
    return data_all

def collect_data_1d(data,nptype,mpitype,comm = MPI.COMM_WORLD,root=0):

    data_all = collect_data(data,1,nptype,mpitype,comm,root=root)
    if not data_all is None:
        data_all = data_all.reshape((-1,))    
    return data_all

def collect_strings(s_list,comm = MPI.COMM_WORLD,root=0):
    rank = comm.Get_rank()
    s_array = [np.array([ch for ch in s]) for s in s_list]
    s_array_all = collect_arrays(s_array,1,'S',MPI.UNSIGNED_CHAR,comm,root=root)
    if rank == 0:
        s_all = [''.join(s.reshape((-1,))) for s in s_array_all]
    else:
        s_all = None
    return s_all

def collect_array_(data_array,nbr_col,nptype,mpitype,comm = MPI.COMM_WORLD,root=0):
    rank = comm.Get_rank()
    if rank == 0:
        nbr_row = np.empty(comm.Get_size(),dtype = 'i')
    else:
        nbr_row = 0
    comm.Gather(sendbuf=[np.array(data_array.shape[0],dtype='i'), MPI.INT], recvbuf=[nbr_row, MPI.INT], root=root)
    counts = nbr_row*nbr_col
    #print "counts = {} at rank {}".format(counts,rank)    
    
    if rank == 0:
        data_all_array = np.empty((sum(nbr_row),nbr_col),dtype = nptype)
    else:
        data_all_array = None
    comm.Gatherv(sendbuf=[data_array,mpitype],recvbuf=[data_all_array,(counts,None),mpitype],root=root)
    return data_all_array
