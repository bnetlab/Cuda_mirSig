import numpy as np
import pandas as pd

def borda (mats, n, nv):
    
    for v in range (nv):
        mats[v] = mats[v].argsort(axis = None)
        
    borda_points = np.zeros((n*n))
    for i in range(n*n):
        # i is the current rank star between 0-15
        # rank pint is i/highest rank possible (n*n-1)
        
        for v in range (nv):
            
            borda_points[(mats[v])[i]] = borda_points[(mats[v])[i]] + (i/(n*n-1))
        
         
    return ((borda_points/nv).reshape(n,n))