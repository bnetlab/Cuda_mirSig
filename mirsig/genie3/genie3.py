from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import *
from joblib import Parallel, delayed
from collections import defaultdict

class BatchCompletionCallBack(object):
  completed = defaultdict(int)
  total = 0
  def __init__(self, time, index, parallel):
    self.index = index
    self.parallel = parallel


  def __call__(self, index):
    BatchCompletionCallBack.completed[self.parallel] += 1
    sys.stdout.write("\r%d%%" % ((BatchCompletionCallBack.completed[self.parallel])/self.total*100))
    sys.stdout.flush()
    if self.parallel._original_iterator is not None:
      self.parallel.dispatch_next()


import joblib.parallel
joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack


total = 0 ## numVars
current = 0 ## current target

def genie_single(expr_data, target_idx, n_trees = 1000, msp = 2, n_jobs = 1):
    global current
    global total
    current +=1
#     print (target_idx)
    y = expr_data[:,target_idx]
    # remove targe gene from perdictors
    X = np.delete(expr_data, target_idx, 1)
    
    #normalise target gene
    y = y/np.std(y)
#     start = time.time()
    forest = RandomForestRegressor(n_estimators=n_trees, max_features="sqrt",
                                       random_state=32, n_jobs= n_jobs,  min_samples_split = 2)
    forest.fit(X,y)
#     end = time.time()
    importances = forest.feature_importances_
    # construct out put
    output = np.empty((expr_data.shape[1]))
    j = 0
    for i in range (expr_data.shape[1]):
        if i == target_idx:
            output[i]=0
        else:
            output[i] = importances[j]
            j+=1
    # print ("TargetIdx : %d\t time : %d" % (target_idx, (end-start)) )


    return output

#parallel for each varibale
def doSinglePar(expr_data,target, n_trees, pranks):
#     print (target)
    return genie_single(expr_data.values, target, n_trees = n_trees)
#     def genie_single(expr_data, target_idx, n_trees = 60, msp = 2):



def parallel_genie(expr_data, n_trees = 650, n_jobs = 20):
    global total
    '''Computation of tree-based scores for all putative regulatory links.

    Parameters
    ----------

    expr_data: Pandas DataFrame
        Array containing gene expression values. Each row corresponds to a condition and each column corresponds to a gene.

    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data, containing the names of the genes. The i-th item of gene_names must correspond to the i-th column of expr_data.
        default: None



    Returns
    -------

    Pandas DataFram for ranked results
    '''
    numSamples = expr_data.shape[0]
    numVars = expr_data.shape[1]
    BatchCompletionCallBack.total = numVars
    joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack
    pranks = np.zeros((numVars,numVars))
    tmp  = Parallel(n_jobs= n_jobs)(delayed(doSinglePar)
                                    (expr_data,target, n_trees, pranks)
                                        for target in range (0,numVars))
    for i in range (numVars):
        pranks[i,:] = tmp[i]
    #transpose to make "ranking of the genese as potential regulators of target"
    pranks = pranks.T
    presults = pd.DataFrame(pranks, columns = expr_data.columns, index = expr_data.columns)
    return presults