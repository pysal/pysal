import pysal as ps
import numpy as np
from numpy import linalg as la
import scipy.stats as st

class BaseGWR(object):
    def __init__(self, y, X, W):
        self.y = y
        self.n = y.shape[0]
        self.p = X.shape[-1]
        self.X = X
        self.W = W
        self.vm = NotImplementedError
        self.betas = []
        self.yhats = []
        
        for i in range(self.n):
            Wuv = np.diag(np.sqrt(self.W.full()[0][i,:]))
            xtw = np.dot(self.X.T, Wuv)
            xtwx = np.dot(xtw, self.X)
            xtwx_inv = la.inv(xtwx)
            xtwx_inv_xtw = np.dot(xtwx_inv, xtw)
            beta_hats = np.dot(xtwx_inv_xtw, self.y)
            self.betas.append(beta_hats.T)
            yhat = np.dot(self.X[i,:], beta_hats)
            self.yhats.append(yhat[0])
        self.betas = np.array(self.betas).flatten()
        self.yhats = np.array(self.yhats).reshape(self.y.shape)
            
        self.SSE = np.sum((self.y - self.yhats)**2)
        self.MSE = self.SSE / float(self.n)
        self.SST = np.sum((self.y - np.mean(self.y))**2)
        self.SSR = self.SST - self.SSE
        self.R2 = self.SSR/self.SST
        
        self.MSR = self.SSR / self.p
        self.MST = self.SST / (self.n - self.p - 1)
        f_stat = self.MSR/self.MST
        self.f_stat = (f_stat, st.f.sf(f_stat, self.p-1, self.n-k))
        
    def __repr__(self):
        return "This is a GWR model"


if __name__ == "__main__":
    data = ps.open(ps.examples.get_path('columbus.dbf'))
    W_rook = ps.open(ps.examples.get_path('columbus.gal'))
    W_kernel = ps.kernelW_from_shapefile(
                ps.examples.get_path('columbus.shp'),
                3, function='gaussian', diagonal=True)
    W_kernel_floatdiag = ps.kernelW_from_shapefile(
                ps.examples.get_path('columbus.shp'),
                3, function='gaussian', diagonal=False) 
   
    y = data.by_col_array(['HOVAL'])
    X1, X2 = data.by_col_array(['INC']), data.by_col_array(['CRIME'])

    X = np.hstack((X1,X2))

    mod1 = BaseGWR(y,X,W_kernel)
    mod2 = BaseGWR(y,X,W_kernel_floatdiag)
