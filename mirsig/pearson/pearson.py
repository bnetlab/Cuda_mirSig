#import tensorflow as tf 

# In[2]:
import numpy as np
import pandas as pd
import dask.array as da
#from tensorflow.python.client import timeline
import dask.dataframe as dd
from mirsig.util import utility as ut
# or from ..util import utility as ut



def pearson_df(df):
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
    panda_cr = df.corr(method='pearson')
    np.fill_diagonal(panda_cr.values, 0)
    return panda_cr


