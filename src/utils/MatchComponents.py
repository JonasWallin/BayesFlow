import numpy as np
import bhat
from flow_match import flow_match

import matplotlib.pyplot as plt


class SampleComponents(object):

    def __init__(self, components=None, j=None, act_thr=0.05):
        self.j = j
        if not components is None:
            self.K = components.K
            if j is not None:
                self.ks = np.nonzero(components.active_komp[j, :]
                                     > act_thr)[0]
                self.mus = [components.mupers[j, k, :] for k in self.ks]
                self.Sigmas = [components.Sigmapers[j, k, :, :] for k in self.ks]
                self.p = [components.p[j, k] for k in self.ks]
            else:
                self.ks = range(self.K)
                self.mus = [components.mulat[k, :] for k in self.ks]
                self.Sigmas = [components.Sigmalat[k, :, :] for k in self.ks]
                self.p = [np.mean(components.p[:, k]) for k in self.ks]
        else:
            self.clear()

    def clear(self):
        self.K = 0
        self.mus = []
        self.Sigmas = []
        self.p = []
        self.ks = []

    def get_component(self, k):
        i = self.ks.index(k)
        return self.mus[i], self.Sigmas[i], self.p[i]

    def remove_component(self, k):
        i = self.ks.index(k)
        self.ks.pop(i)
        self.mus.pop(i)
        self.Sigmas.pop(i)
        self.p.pop(i)
        self.K -= 1

    def append_component(self, mu, Sigma, p, k=None):
        if k is None:
            if len(self.ks) > 0:
                k = max(self.ks) + 1
            else:
                k = 0
        self.K += 1
        self.mus.append(mu)
        self.Sigmas.append(Sigma)
        self.p.append(p)
        self.ks.append(k)

    def merge_components(self, ks):
        comps = [self.get_component(k) for k in ks]
        mus = [comps[0] for comp in comps]
        Sigmas = [comps[1] for comp in comps]
        p = [comps[2] for comp in comps]

        new_p = sum(p)
        new_mu = sum([p[i]*mu for i, mu in enumerate(mus)])/new_p
        new_Sigma = sum([p[i]*Sigma for i, Sigma in enumerate(Sigmas)])/new_p

        for k in ks:
            self.remove_component(k)

        self.append_component(new_mu, new_Sigma, new_p, ks[0])
        return ks[0]

    def relabel(self, min_k):
        self.ks = range(min_k, min_k+self.K)

    def concatenate(self, samp_comp):
        for k in samp_comp.ks:
            self.append_component(samp_comp.get_component(k))

    def move_unmatched_to_matched(self):
        self.concatenate(self.unmatched_comp)
        self.unmatched_comp.clear()

    def bhattacharyya_distance_to(self, samp_comp):
        '''
            Get Bhattacharyya distance between all own components
            and all components in samp_comp.

            Returns a (self.K x samp_comp.K) matrix where element (k, l)
            is the distance from own k to samp_comp component l.
        '''
        bhd = np.empty((self.K, samp_comp.K))
        for k in range(self.K):
            for l in range(self.K):
                bhd[k, l] = bhat.bhattacharyya_distance(
                    self.mus[k], self.Sigmas[k],
                    samp_comp.mus[l], samp_comp.Sigmas[l])
        return bhd

    def match_to(self, latent, lamb):
        bhd = self.bhattacharyya_distance_to(latent)
        match12, match21, matching_cost, unmatch_penalty = flow_match(bhd, lamb)

        old_ks = self.ks[:]
        self.unmatched_comp = SampleComponents(j=self.j)
        new_ks_dict = {}
        merged_latent = []

        for i, ind in enumerate(match12):
            k = old_ks[i]
            if len(ind) == 0:  # sample component k is unmatched
                self.unmatched_comp.append_component(self.get_component(k))
                self.remove_component(k)
            elif len(ind) == 1:
                # sample component k matches to exactly one latent
                # component, ind[0]
                if ind[0] in merged_latent:
                    continue
                i_match = match21[ind[0]]
                    # find all components that match latent component ind[0]
                ks = [old_ks[i] for i in i_match]
                k = self.merge_components(ks)
                new_ks_dict[k] = latent.ks[ind[0]]
                merged_latent.append(ind[0])
            else:
                # sample component k matches to more than one latent component.
                # match to closest latent component of these.
                new_ks_dict[k] = latent.ks[np.argmin(bhd[i, ind])]

        self.ks = [new_ks_dict[kk] for kk in self.ks]


def match_components(comps):
    samp_comps = [SampleComponents(comps, j) for j in comps.j]
    latent = SampleComponents(comps)

    for sc in samp_comps:
        sc.match_to(latent)

    new_latent = samp_comps[0].unmatched_comp.relabel(latent.K)
    samp_comps[0].move_unmatched_to_matched()

    for sc in samp_comps[1:]:
        sc.unmatched_comp.match_to(new_latent)
        sc.unmatched_comp.unmatched_comp.relabel(latent.K+new_latent.K)
        new_latent.concatenate(sc.unmatched_comp.unmatched_comp)

        sc.unmatched_comp.move_unmatched_to_matched()
        sc.move_unmatched_to_matched()

    latent.concatenate(new_latent)

    return samp_comps, latent


def center_plot(samp_comps, latent, fig=None, totplots=1, plotnbr=1,
                yscale=False, ks=None):
        '''
            The centers of all components, mu, are plotted along one dimension.
        '''
        if fig is None:
            fig = plt.figure()

        d = latent.d

        ks_ = latent.ks[:]
        if not ks is None:
            ks_ = list(set(ks_).intersection(ks))
        ks_.sort(key=lambda k: latent.get_component(k)[2])  # sort by size

        S = len(ks_)
        nbr_cols = 2*totplots-1
        col_start = 2*(plotnbr-1)

        for s, k in enumerate(ks_):
            ax = fig.add_subplot(S, nbr_cols, s*nbr_cols + col_start+1)
            for sc in samp_comps:
                ax.plot(range(d), sc.get_component(k)[0], color=(0, 0, 1, 0.5))
            ax.plot(range(d), latent.get_component(k)[0], color=(0, 0, 0))
            ax.plot([0, d-1], [.5, .5], color='grey')
            if s == S-1:
                ax.axes.xaxis.set_ticks(range(d))
                #ax.set_xticklabels(self.marker_lab)
            else:
                ax.axes.xaxis.set_ticks([])
            if not yscale:
                ax.axes.yaxis.set_ticks([])
                ax.set_ylim(0, 1)
            else:
                ax.axes.yaxis.set_ticks([.2, .8])
                ax.set_ylim(-.1, 1.1)
