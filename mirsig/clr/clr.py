import ctypes as C
import numpy as np
import pandas as pd
import os


# clr calculates z-score of mutual information matrix between each variable (Gene Expression)

# input_mat : NumSamples X NumVar input data
# retrun: NumVars X NumVar weighted matrix





def clr(input_mat, batchSize, threadPerBlock, verbosity=1, numBins=-1):

    # """
    # context likelihood of relatedness algorithm returns the weighted edges of possible
    # interaction network. The process runs on gpu, it's possible to use the blocking 
    # in order to reduce the the load and improve the performance.
    # ----------
    # input_mat : array 
    #     input array of gene expression data samples as rows with type "float32"
    # batchSize : int
    #     size of each batch to be send to the gpu in one dimension
    # threadPerBlock : int
    #     number of threads to be used for each block on GPU
    # numBins : number of beans that is used to estimate the mutual information values use "-1"
    #           for auto
    # verbosity : 1-3 
    # Returns
    # -------
    # Array : inferred network with weighted edges.


    # """
    # print()
    

    if(type(input_mat) is pd.core.frame.DataFrame):
        input_mat = input_mat.values


    _testlib = C.CDLL(os.path.dirname(os.path.realpath(__file__))+'/template.so')
    _testlib.clr.argtypes = (C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, np.ctypeslib.ndpointer(
        dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32))
    _testlib.clr.restype = C.c_void_p

    numSamples = input_mat.shape[0]
    numVars = input_mat.shape[1]

    results = np.zeros(numVars*numVars, dtype=np.float32)

    h_data = input_mat.copy()
    h_data = h_data.astype(np.float32)
    h_data = h_data.flatten(order='F')

    _testlib.clr(numSamples, numVars, numBins, batchSize,
                 threadPerBlock, verbosity, h_data, results)

    return np.reshape(results, (numVars, numVars), order='F')


# h_data = np.random.randn(40, 100)
# print(h_data)
# h_data = h_data.astype(np.float32)


# print(clr(h_data, 10, 16))
