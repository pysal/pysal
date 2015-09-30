import pysal as ps
import numpy as np
from numpy import linalg as la
import scipy.stats as st

class BaseML_GWLag(RegressionPropsY, RegressionPropsVM):
    def __init__(self, y, x, w, method='full', epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.y = y
        self.x = x
        self.n, self.k = self.x.shape
        self.method = method
        self.epsilon = epsilon
        self.ylags = []
        self.rhos = []
        self.logls = []
        self.betas = []
        self.predys = []
        self.pred_es = []
        self.e_preds = []
        self.us = []
        self.vms = []
        for i in w.shape[0]:
            Wvect = w[i,:]
            Wmat = np.diag(Wvect)
            ylag = np.sum(y * Wvect) / Wvect.sum()
            self.ylags.append(ylag)

            # b0, b1, e0 and e1
            xtx = spdot(self.x.T, self.x)
            xtxi = la.inv(xtx)
            xty = spdot(self.x.T, self.y)
            xtyl = spdot(self.x.T, ylag)
            b0 = np.dot(xtxi, xty)
            b1 = np.dot(xtxi, xtyl)
            e0 = self.y - spdot(x, b0)
            e1 = ylag - spdot(x, b1)
            methodML = method.upper()
            # call minimizer using concentrated log-likelihood to get rho
            if methodML in ['FULL', 'ORD']:
                if methodML == 'FULL':
                    res = minimize_scalar(lag_c_loglik, 0.0, bounds=(-1.0, 1.0),
                                          args=(
                                              self.n, e0, e1, W), method='bounded',
                                          tol=epsilon)
                elif methodML == 'ORD':
                    # check on symmetry structure
                    if w.asymmetry(intrinsic=False) == []:
                        ww = symmetrize(w)
                        WW = ww.todense()
                        evals = la.eigvalsh(WW)
                    else:
                        evals = la.eigvals(W)
                    res = minimize_scalar(lag_c_loglik_ord, 0.0, bounds=(-1.0, 1.0),
                                          args=(
                                              self.n, e0, e1, evals), method='bounded',
                                          tol=epsilon)
            else:
                # program will crash, need to catch
                print("{0} is an unsupported method".format(methodML))
                self = None
                return
            self.rhos.append(res.x[0][0])
            rho = self.rhos[i]

            # compute full log-likelihood, including constants
            ln2pi = np.log(2.0 * np.pi)
            llik = -res.fun - self.n / 2.0 * ln2pi - self.n / 2.0
            self.loglls.append(llik[0][0])

            # b, residuals and predicted values

            b = b0 - rho * b1
            self.betas.append(np.vstack((b,self.rhos[i])))  # rho added as last coefficient
            beta = self.betas[i]
            self.us.append(e0 - rho * e1)
            u = self.us[i]
            predy = self.y - self.u
            self.predys.append(predy)

            xb = spdot(x, b)

            predy_e = inverse_prod(
                w.sparse, xb, self.rho, inv_method="power_exp", threshold=epsilon)
            self.predy_es.append(predy_e)
            e_pred = self.y - self.predy_e
            self.e_preds.append(e_pred)

            # residual variance
            #self._cache = {}
            #self.sig2 = self.sig2n  # no allowance for division by n-k

            # information matrix
            a = -self.rho * W
            np.fill_diagonal(a, 1.0)
            ai = la.inv(a)
            wai = np.dot(W, ai)
            tr1 = np.trace(wai)

            wai2 = np.dot(wai, wai)
            tr2 = np.trace(wai2)

            waiTwai = np.dot(wai.T, wai)
            tr3 = np.trace(waiTwai)

            wpredy = ps.lag_spatial(w, self.predy_e)
            wpyTwpy = np.dot(wpredy.T, wpredy)
            xTwpy = spdot(x.T, wpredy)

            # order of variables is beta, rho, sigma2

            v1 = np.vstack(
                (xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((1, self.k))))
            v2 = np.vstack(
                (xTwpy / self.sig2, tr2 + tr3 + wpyTwpy / self.sig2, tr1 / self.sig2))
            v3 = np.vstack(
                (np.zeros((self.k, 1)), tr1 / self.sig2, self.n / (2.0 * self.sig2 ** 2)))

            v = np.hstack((v1, v2, v3))

            vm1 = la.inv(v)  # vm1 includes variance for sigma2
            self.vms.append(vm1[:-1, :-1])  # vm is for coefficients only


