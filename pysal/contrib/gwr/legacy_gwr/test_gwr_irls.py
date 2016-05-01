def _get_Betas(self, g_ey, fMatrix):
    """
    get Beta estimations
        
    Methods: p189, Iteratively Reweighted Least Squares (IRLS), Fotheringham, Brunsdon and Charlton (2002)
             Tomoki et al. (2005), (18)-(25)  
    """        
    nloops = len(self.kernel.w.keys())
    nIter = [0]*nloops       
	
    Betas = np.zeros(shape=(self.nObs,self.nVars)) 
    zs = np.ones(shape=(self.nObs,1)) 
    vs = np.zeros(shape=(self.nObs,1)) 
    ws = np.zeros(shape=(self.nObs,1))
        
    # 1 get initial Betas estimation using global model
    if self.Betas_ini is None:
        Betas_ini = get_Betas_ini(g_ey-self.y_fix,self.x) #  
    else:
        Betas_ini = self.Betas_ini
        
    # estimate Betas for each observation
    for i in range(nloops):    # self.nObs
        diff = 1e6 
        Betas_ini
        v = np.dot(self.x, Betas_old) 
        
        while diff > self.tol and nIter[i] < self.maxIter:
            nIter[i] += 1 
            z, w_new = get_link[self.mType](v, self.y, self.offset, self.y_fix)                 
               
            arr_w = np.reshape(np.array(self.kernel.w[i]), (-1, 1)) * w_new #w_new[i]
            w_i = np.sqrt(arr_w)
            
            xw = self.x * w_i            
            
            xtwx = np.dot(xw.T, xw)
            xtwx_inv = la.inv(xtwx)              
            xtw = (self.x * arr_w).T
            xtwx_inv_xtw = np.dot(xtwx_inv, xtw)

            
            Betas_new = np.dot(xtwx_inv_xtw, z)          
                
                
            # 4 determin convergence or number of iteration
            v_new = np.dot(self.x, Betas_new)  #np.reshape(np.sum(self.x * Betas_new, axis=1), (-1,1))         
                            
                
            if self.mType == 0:
                diff = 0 # Gaussian model
            else:
                if nIter[i] > 1:
                    diff = np.min(abs(Betas_new - Betas_old)) # minimum residual
    # 5 update variables
            v = v_new
            Betas_old = Betas_new 	
		
        Betas[i,:] = Betas_new.T
        zs[i] = z[i]
        vs[i] = v[i]
        ws[i] = w_new[i]
            
    return Betas, zs, ws, nIter, vs      
