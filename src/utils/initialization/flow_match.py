# try:
#     from rpy2.robjects.packages import importr
#     from rpy2.rinterface import RRuntimeError
#     from rpy2 import robjects
# except:
#     pass


def flow_match(dist_matrix, lambd):

    try:
        flowMatch = importr('flowMatch')
    except RRuntimeError as e:
        print("You need to install the R Bioconductor package flowMatch to use flow_match")
        raise e

    K, L = dist_matrix.shape
    
    if K == 0:
        match12 = []
        match21 = [[] for l in range(L)]
        matching_cost = 0
        unmatch_penalty = lambd*L
        return match12, match21, matching_cost, unmatch_penalty
        
    if L == 0:
        match12 = [[] for k in range(K)]
        match21 = []
        matching_cost = 0
        unmatch_penalty = lambd*K
        return match12, match21, matching_cost, unmatch_penalty

    dist_matrix_R = robjects.r.matrix(robjects.FloatVector(
        dist_matrix.ravel(order='F')), nrow=dist_matrix.shape[0])

    mec = flowMatch.match_clusters_dist(dist_matrix_R, lambd)

    match12_R = mec.do_slot('match12')
    match21_R = mec.do_slot('match21')
    match12 = []
    

    
    for k in range(K):
        match12.append([int(n)-1 for n in list(match12_R[k])])
    match21 = []
    for l in range(L):
        match21.append([int(n)-1 for n in list(match21_R[l])])
    matching_cost = float(mec.do_slot('matching.cost')[0])
    unmatch_penalty = float(mec.do_slot('unmatch.penalty')[0])

    return match12, match21, matching_cost, unmatch_penalty

if __name__ == '__main__':

    K = 3
    L = 4
    lambd = 0.6
    a = 0.2*np.array(range(K*L)).reshape(K, L)
    print("a = {}".format(a))
    a_R = robjects.r.matrix(robjects.FloatVector(a.ravel(order='F')), nrow=a.shape[0])
    print("a_R = {}".format(a_R))

    print("flow_match(a, lambd) = {}".format(flow_match(a, lambd)))
