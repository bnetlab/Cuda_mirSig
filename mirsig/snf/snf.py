import numpy as np

def SNF(wall, K, t):
    lw = len(wall)
    def normalize(X):
        return (X.T/np.sum(X,axis=1)).T # devide each row by its sum
    for i in range(lw):
        wall[i] = wall[i].copy()
        wall[i] = meanNormalise(wall[i])
    newW = [None]*lw
    nextW = [None]*lw
    for i in range(lw):
        # full matrix
        wall[i] = dominateset((wall[i]+wall[i].T)/2, 2*K) # not in paper
        
        wall[i] = wall[i]+wall[i].T
    
    for i in range (lw):
        # sparse
        newW[i] = dominateset((wall[i]+wall[i].T)/2, K) 
    
    for i in range(t):
        for j in range (lw):
            sumWJ = np.zeros((wall[j].shape[0],wall[j].shape[1]))
            for k in range (lw):
                if (k!= j):
                    sumWJ = sumWJ+wall[k]
            
            nextW[j] = np.dot(
                            np.dot(newW[j],
                                (0.95*sumWJ/(lw-1)+0.05*np.eye(wall[j].shape[0]))),
                            newW[j].T)
        # print(wall)    
        for j in range(lw):
            nextW[j] = normalize(nextW[j])
            
            wall[j] = normalize(nextW[j]+np.eye(wall[j].shape[0]))
            wall[j] = (wall[j]+wall[j].T)/2
        #print(wall)
    
    w = np.zeros((wall[j].shape[0],wall[j].shape[1]))
    for i in range(lw):
        w = w + wall[i]
    w = w/lw
    w = (w+w.T)/2
    return (w)
                            

def meanNormalise(XX):
    return ((XX.T-np.median(XX,axis=1))/(XX.std(axis=1))).T
    #  return (XX.T/np.sum(XX,axis=1)).T
def dominateset(XX, kk = 20):
    
    XXc = XX.copy()
    
    def zero (x, kk):
        s = np.argsort(x)
        
        x[s[0:-kk]] = 0
        
        return(x)
    def normalize(X):
        return (X.T/np.sum(X,axis=1)).T # devide each row by its sum
    
    A = np.zeros((XXc.shape[0], XXc.shape[1]))
    for i in range(XXc.shape[0]):
        A[i,:] = zero(XXc[i,:],kk)
    return (normalize(A))