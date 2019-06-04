
import numpy as np
import pandas as pd
import dask.array as da
#from tensorflow.python.client import timeline
import dask.dataframe as dd
import scipy.stats as st
from mirsig.util import utility as ut
# or from ..util import utility as ut



def spearman_df(df):
    """ 
    Calculates pearson collrelation on input and returns ranked matrix
    
    Parameters
    ----------
    df : array, shape = [n]
        x coordinates.
    
    Returns
    -------
    df : pandas array, 

    """
    
    rho, pval = st.spearmanr(df.values)
    np.fill_diagonal(rho, 0)
    panda_cr = pd.DataFrame(rho, index= df.columns, columns= df.columns)
    return panda_cr
