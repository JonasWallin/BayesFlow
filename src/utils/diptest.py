from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
# try:
#     from rpy2.robjects.packages import importr, data
#     from rpy2.rinterface import RRuntimeError
# except Exception as e:
#     pass
    #print("{} --- will not be able to transform dip values to p-value".format(e))


def dip_from_cdf(xF, yF, plotting=False, verbose=False, eps=1e-12):
    '''
        Dip computed as distance between empirical distribution function (EDF) and
        cumulative distribution function for the unimodal distribution with
        smallest such distance. The optimal unimodal distribution is found by
        the algorithm presented in

            Hartigan (1985): Computation of the dip statistic to test for
            unimodaliy. Applied Statistics, vol. 34, no. 3

        If the plotting option is enabled the optimal unimodal distribution
        function is plotted along with (xF, yF-dip) and (xF, yF+dip)

        xF  -   x-coordinates for EDF
        yF  -   y-coordinates for EDF

    '''

    ## TODO! Preprocess xF and yF so that yF increasing and xF does
    ## not have more than two copies of each x-value.

    if (xF[1:]-xF[:-1] < -eps).any():
        raise ValueError('Need sorted x-values to compute dip')
    if (yF[1:]-yF[:-1] < -eps).any():
        raise ValueError('Need sorted y-values to compute dip')

    if plotting:
        Nplot = 5
        bfig = plt.figure(figsize=(12, 3))
        i = 1  # plot index

    D = 0  # lower bound for dip*2

    # [L, U] is interval where we still need to find unimodal function,
    # the modal interval
    L = 0
    U = len(xF) - 1

    # iGfin are the indices of xF where the optimal unimodal distribution is greatest
    # convex minorant to (xF, yF+dip)
    # iHfin are the indices of xF where the optimal unimodal distribution is least
    # concave majorant to (xF, yF-dip)
    iGfin = L
    iHfin = U

    while 1:

        iGG = greatest_convex_minorant_sorted(xF[L:(U+1)], yF[L:(U+1)])
        iHH = least_concave_majorant_sorted(xF[L:(U+1)], yF[L:(U+1)])
        iG = np.arange(L, U+1)[iGG]
        iH = np.arange(L, U+1)[iHH]

        # Find largest difference between GCM and LCM.
        hipl = lin_interpol_sorted(xF[iG], xF[iH], yF[iH])
        gipl = lin_interpol_sorted(xF[iH], xF[iG], yF[iG])
        gdiff = hipl - yF[iG]
        hdiff = yF[iH] - gipl
        imaxdiffg = np.argmax(gdiff)
        imaxdiffh = np.argmax(hdiff)
        d = max(gdiff[imaxdiffg], hdiff[imaxdiffh])

        # Plot current GCM and LCM.
        if plotting:
            if i > Nplot:
                bfig = plt.figure(figsize=(12, 3))
                i = 1
            bax = bfig.add_subplot(1, Nplot, i)
            bax.plot(xF, yF, color='red')
            bax.plot(xF, yF-d/2, color='black')
            bax.plot(xF, yF+d/2, color='black')
            bax.plot(xF[iG], yF[iG]+d/2, color='blue')
            bax.plot(xF[iH], yF[iH]-d/2, color='blue')

        if d <= D:
            if verbose:
                print("Difference in modal interval smaller than current dip")
            break

        # Find new modal interval so that largest difference is at endpoint
        # and set d to largest distance between current GCM and LCM.
        if gdiff[imaxdiffg] > hdiff[imaxdiffh]:
            L0 = iG[imaxdiffg]
            U0 = iH[iH >= L0][0]
        else:
            U0 = iH[imaxdiffh]
            L0 = iG[iG <= U0][-1]
        # Add points outside the modal interval to the final GCM and LCM.
        iGfin = np.hstack([iGfin, iG[(iG <= L0)*(iG > L)]])
        iHfin = np.hstack([iH[(iH >= U0)*(iH < U)], iHfin])

        # Plot new modal interval
        if plotting:
            ymin, ymax = bax.get_ylim()
            bax.axvline(xF[L0], ymin, ymax, color='orange')
            bax.axvline(xF[U0], ymin, ymax, color='red')
            bax.set_xlim(xF[L]-.1*(xF[U]-xF[L]), xF[U]+.1*(xF[U]-xF[L]))

        if xF[U0]-xF[L0] < eps:
            if verbose:
                print("Modal interval zero length")
            break

        # Compute new lower bound for dip*2
        # i.e. largest difference outside modal interval
        gipl = lin_interpol_sorted(xF[L:(L0+1)], xF[iG], yF[iG])
        D = max(D, np.amax(yF[L:(L0+1)] - gipl))
        hipl = lin_interpol_sorted(xF[U0:(U+1)], xF[iH], yF[iH])
        D = max(D, np.amax(hipl - yF[U0:(U+1)]))

        if plotting:
            mxpt = np.argmax(yF[L:(L0+1)] - gipl)
            bax.plot([xF[L:][mxpt], xF[L:][mxpt]], [yF[L:][mxpt]+d/2, gipl[mxpt]+d/2], '+', color='red')
            mxpt = np.argmax(hipl - yF[U0:(U+1)])
            bax.plot([xF[U0:][mxpt], xF[U0:][mxpt]], [yF[U0:][mxpt]-d/2, hipl[mxpt]-d/2], '+', color='red')
            i += 1

        # Change modal interval
        L = L0
        U = U0

        if d <= D:
            if verbose:
                print("Difference in modal interval smaller than new dip")
            break

    if plotting:

        # Add modal interval to figure
        bax.axvline(xF[L0], ymin, ymax, color='green', linestyle='dashed')
        bax.axvline(xF[U0], ymin, ymax, color='green', linestyle='dashed')

        ## Plot unimodal function (not distribution function)
        bfig = plt.figure()
        bax = bfig.add_subplot(1, 1, 1)
        bax.plot(xF, yF, color='red')
        bax.plot(xF, yF-D/2, color='black')
        bax.plot(xF, yF+D/2, color='black')

        # Find string position in modal interval
        print("iHfin = {}".format(iHfin))
        print("xF.shape = {}".format(xF.shape))
        iM = np.arange(iGfin[-1], iHfin[0]+1)
        yM_lower = yF[iM]-D/2
        yM_lower[0] = yF[iM[0]]+D/2
        iMM_concave = least_concave_majorant_sorted(xF[iM], yM_lower)
        iM_concave = iM[iMM_concave]
        #bax.plot(xF[iM], yM_lower, color='orange')
        #bax.plot(xF[iM_concave], yM_lower[iMM_concave], color='red')
        lcm_ipl = lin_interpol_sorted(xF[iM], xF[iM_concave], yM_lower[iMM_concave])
        try:
            mode = iM[np.nonzero(lcm_ipl > yF[iM]+D/2)[0][-1]]
            #bax.axvline(xF[mode], color='green', linestyle='dashed')
        except IndexError:
            iM_convex = np.zeros(0, dtype='i')
        else:
            after_mode = iM_concave > mode
            iM_concave = iM_concave[after_mode]
            iMM_concave = iMM_concave[after_mode]
            iM = iM[iM <= mode]
            iM_convex = iM[greatest_convex_minorant_sorted(xF[iM], yF[iM])]

        bax.plot(xF[np.hstack([iGfin, iM_convex, iM_concave, iHfin])],
                 np.hstack([yF[iGfin] + D/2, yF[iM_convex] + D/2,
                            yM_lower[iMM_concave], yF[iHfin] - D/2]), color='blue')
        #bax.plot(xF[iM], yM_lower, color='orange')

        ## Plot unimodal distribution function
        bfig = plt.figure()
        bax = bfig.add_subplot(1, 1, 1)
        bax.plot(xF, yF, color='red')
        bax.plot(xF, yF-D/2, color='black')
        bax.plot(xF, yF+D/2, color='black')

        # Find string position in modal interval
        print("iHfin = {}".format(iHfin))
        print("xF.shape = {}".format(xF.shape))
        iM = np.arange(iGfin[-1], iHfin[0]+1)
        yM_lower = yF[iM]-D/2
        yM_lower[0] = yF[iM[0]]+D/2
        iMM_concave = least_concave_majorant_sorted(xF[iM], yM_lower)
        iM_concave = iM[iMM_concave]
        #bax.plot(xF[iM], yM_lower, color='orange')
        #bax.plot(xF[iM_concave], yM_lower[iMM_concave], color='red')
        lcm_ipl = lin_interpol_sorted(xF[iM], xF[iM_concave], yM_lower[iMM_concave])
        try:
            mode = iM[np.nonzero(lcm_ipl > yF[iM]+D/2)[0][-1]]
            #bax.axvline(xF[mode], color='green', linestyle='dashed')
        except IndexError:
            iM_convex = np.zeros(0, dtype='i')
        else:
            after_mode = iM_concave > mode
            iM_concave = iM_concave[after_mode]
            iMM_concave = iMM_concave[after_mode]
            iM = iM[iM <= mode]
            iM_convex = iM[greatest_convex_minorant_sorted(xF[iM], yF[iM])]

        bax.plot(xF[np.hstack([iGfin, iM_convex, iM_concave, iHfin])],
                 np.hstack([yF[iGfin[0]], yF[iGfin[1:]] + D/2, yF[iM_convex] + D/2,
                            yM_lower[iMM_concave], yF[iHfin[:-1]] - D/2,
                            yF[iHfin[-1]]]), color='blue')
        #bax.plot(xF[iM], yM_lower, color='orange')
        plt.show()

    return D/2


def dip_pval_tabinterpol(dip, N):
    '''
        Tablulated values are obtained from R package 'diptest', which is
        installed if it is not done previsously. Alternatively, it can be
        installed from within R by "install.packages('diptest')"

        dip     -   dip value computed from dip_from_cdf
        N       -   number of observations
    '''
    if np.isnan(N) or N < 10:
        return np.nan
    try:
        diptest = importr('diptest')
    except RRuntimeError:
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('diptest')
        diptest = importr('diptest')

    qDiptab = data(diptest).fetch('qDiptab')['qDiptab']
    diptable = np.array(qDiptab)
    ps = np.array(qDiptab.colnames).astype(float)
    Ns = np.array(qDiptab.rownames).astype(int)

    if N >= Ns[-1]:
        dip = transform_dip_to_other_nbr_pts(dip, N, Ns[-1]-0.1)
        N = Ns[-1]-0.1

    iNlow = np.nonzero(Ns < N)[0][-1]
    qN = (N-Ns[iNlow])/(Ns[iNlow+1]-Ns[iNlow])
    dip_sqrtN = np.sqrt(N)*dip
    dip_interpol_sqrtN = (
        np.sqrt(Ns[iNlow])*diptable[iNlow, :] + qN*(
            np.sqrt(Ns[iNlow+1])*diptable[iNlow+1, :]-np.sqrt(Ns[iNlow])*diptable[iNlow, :]))

    if not (dip_interpol_sqrtN < dip_sqrtN).any():
        return 1

    iplow = np.nonzero(dip_interpol_sqrtN < dip_sqrtN)[0][-1]
    if iplow == len(dip_interpol_sqrtN) - 1:
        return 0

    qp = (dip_sqrtN-dip_interpol_sqrtN[iplow])/(dip_interpol_sqrtN[iplow+1]-dip_interpol_sqrtN[iplow])
    p_interpol = ps[iplow] + qp*(ps[iplow+1]-ps[iplow])

    return 1 - p_interpol


def transform_dip_to_other_nbr_pts(dip_n, n, m):
    dip_m = np.sqrt(n/m)*dip_n
    return dip_m


def cum_distr(data, w):
    eps = 1e-10
    data_ord = np.argsort(data)
    data_sort = data[data_ord]
    w_sort = w[data_ord]
    data_sort, indices = unique(data_sort, return_index=True, eps=eps, is_sorted=True)
    if len(indices) < len(data_ord):
        w_unique = np.zeros(len(indices))
        for i in range(len(indices)-1):
            w_unique[i] = np.sum(w_sort[indices[i]:indices[i+1]])
        w_unique[-1] = np.sum(w_sort[indices[-1]:])
        w_sort = w_unique
    wcum = np.cumsum(w_sort)
    wcum /= wcum[-1]

    N = len(data_sort)
    x = np.empty(2*N)
    x[2*np.arange(N)] = data_sort
    x[2*np.arange(N)+1] = data_sort
    y = np.empty(2*N)
    y[0] = 0
    y[2*np.arange(N)+1] = wcum
    y[2*np.arange(N-1)+2] = wcum[:-1]
    return x, y


def lin_interpol(xquery, x, y):
    xq_ord = np.argsort(xquery)
    xord = np.argsort(x)
    values = lin_interpol_sorted(xquery[xq_ord], x[xord], y[xord])
    return values[np.argsort(xq_ord)]


def lin_interpol_sorted(xquery, x, y, eps=1e-10):
    x, i = unique(x, return_index=True, eps=eps, is_sorted=True)
    y = y[i]
    if len(x) == 1:
        if np.abs(x - xquery).all() < eps:
            return y*np.ones(len(xquery))
        else:
            raise ValueError('interpolation points outside interval')
    i = 0
    j = 1
    if xquery[0] < x[0]-eps:
        raise ValueError('interpolation points outside interval: xquery[0] = {}, x[0] = {}'.format(xquery[0], x[0]))
    values = np.zeros(len(xquery))
    indices = np.zeros(len(xquery))
    while i < len(xquery) and j < len(x):
        if xquery[i] <= x[j]+eps:
            q = (y[j]-y[j-1])/(x[j]-x[j-1])
            values[i] = y[j-1] + q*(xquery[i]-x[j-1])
            indices[i] = x[j-1]
            i += 1
        else:
            j += 1
    if i < len(xquery) - 1:
        raise ValueError('interpolation points outside interval: xquery[-1] = {}, x[-1] = {}'.format(xquery[-1], x[-1]))
    return values


def unique(data, return_index, eps, is_sorted=True):
    if not is_sorted:
        ord = np.argsort(data)
        rank = np.argsort(ord)
        data_sort = data[ord]
    else:
        data_sort = data
    isunique_sort = np.ones(len(data_sort), dtype='bool')
    j = 0
    for i in range(1, len(data_sort)):
        if data_sort[i] - data_sort[j] < eps:
            isunique_sort[i] = False
        else:
            j = i
    if not is_sorted:
        isunique = isunique_sort[rank]
        data_unique = data[isunique]
    else:
        data_unique = data[isunique_sort]

    if not return_index:
        return data_unique

    if not is_sorted:
        ind_unique = np.nonzero(isunique)[0]
    else:
        ind_unique = np.nonzero(isunique_sort)[0]
    return data_unique, ind_unique


def greatest_convex_minorant(x, y):
    i, xnew, negy = least_concave_majorant(x, -y)
    return i, xnew, -negy


def greatest_convex_minorant_sorted(x, y):
    i = least_concave_majorant_sorted(x, -y)
    return i


def least_concave_majorant(x, y, eps=1e-12):

    if (x[1:]-x[:-1] < -eps).any():
        raise ValueError('need sorted x-values to find least concave majorant')

    ind = least_concave_majorant_sorted(x, y, eps)
    ind = np.sort(ind)
    return ind, x[ind], y[ind]


def least_concave_majorant_sorted(x, y, eps=1e-12):
    i = [0]
    icurr = 0
    while icurr < len(x) - 1:
        if np.abs(x[icurr+1]-x[icurr]) > eps:
            q = (y[(icurr+1):]-y[icurr])/(x[(icurr+1):]-x[icurr])
            icurr += 1 + np.argmax(q)
            i.append(icurr)
        elif y[icurr+1] > y[icurr] or icurr == len(x)-2:
            icurr += 1
            i.append(icurr)
        elif np.abs(x[icurr+2]-x[icurr]) > eps:
            q = (y[(icurr+2):]-y[icurr])/(x[(icurr+2):]-x[icurr])
            icurr += 2 + np.argmax(q)
            i.append(icurr)
        else:
            print("x[icurr] = {}, x[icurr+1] = {}, x[icurr+2] = {}".format(x[icurr], x[icurr+1], x[icurr+2]))
            raise ValueError('Maximum two copies of each x-value allowed')

    return np.array(i)

if __name__ == '__main__':
    #seed = np.random.randint(1000)
    for seed in [None, 403, 796]:
        if seed is None:
            dat = np.hstack([np.arange(0, 1, .1), np.arange(2, 3, 0.1)])
        else:
            print("seed = {}".format(seed))
            np.random.seed(seed)
            dat = np.hstack([np.random.randn(10), np.random.randn(10)+2])
        xcum, ycum = cum_distr(dat, np.ones(len(dat))*1./len(dat))
        dip = dip_from_cdf(xcum, ycum, verbose=True, plotting=True)
        print("dip = {}".format(dip))

    for (dip, N, M) in [(0.005, 20000, 50000), (0.01, 2000, 5000), (0.001, 70000, 10000), (0.0005, 1000000, 10000)]:
        print("dip_pval_tabinterpol(dip, N) = {}".format(dip_pval_tabinterpol(dip, N)))
        print("dip_pval_tabinterpol(transform_dip_to_other_nbr_pts(dip, N, M), M) = {}".format(dip_pval_tabinterpol(transform_dip_to_other_nbr_pts(dip, N, M), M)))
