from process import PoissonPointProcess as csr
import numpy as np


def g(pp, intervals=10, dmin=0.0, dmax=None, d=None):
    """
    G function
    """

    # res = pp.G(intervals, dmin, dmax, d)
    if d is None:
        w = pp.max_nnd/intervals
        if dmax:
            w = dmax/intervals
        d = [w*i for i in range(intervals + 2)]
    cdf = [0] * len(d)
    for i, d_i in enumerate(d):
        smaller = [nndi for nndi in pp.nnd if nndi <= d_i]
        cdf[i] = len(smaller)*1./pp.n
    return np.vstack((d, cdf)).T


def g_envelopes(pp, intervals=10, d=None, reps=99, pct=0.05):
    obs = g(pp, intervals=intervals, d=d)
    sim = csr(pp.window, pp.n, reps, asPP=True)
    gs = np.asarray([g(p, d=obs[:, 0]) for p in sim.realizations.values()])
    gs = gs[:, :, -1]
    gs.sort(axis=0)
    low = gs[np.int(reps * pct)]
    high = gs[np.int(reps * (1-pct))]
    x = obs[:, 0]
    gobs = obs[:, 1]
    mean = gs.mean(axis=0)

    return [gs, x, gobs, mean, low, high]
