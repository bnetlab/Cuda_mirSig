
# coding: utf-8

# In[1]:


import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pandas as pd
import dask.array as da
#from tensorflow.python.client import timeline
import dask.dataframe as dd
import scipy.stats as st
import mirsig.util.utility as ut
import mirsig
import re

from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, f1_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# row major
# n = numvars
# nv = number of voters, algorithms
# mats = list of all matricies
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


# # Test

# In[3]:


# dream_df, column_names = ut.get_data("data/dream4/insilico_size100_2_multifactorial.tsv", sep ='\t')[0:2]
# dream_df, column_names = ut.get_data("PD_data_GEO.csv")[0:2]


# In[4]:


dream_df = pd.read_csv("AD_data_GEO.csv", index_col=0)


# In[10]:


dream_df.head()


# In[11]:


column_names = list(dream_df.index)
dream_df = dream_df.T


# In[12]:


column_names


# In[13]:


# run mrnetb befor clr!!
mrnetb_res= mirsig.mrnetb_cuda(dream_df)


# In[14]:


np.save('ADmrnetb', mrnetb_res)


# In[15]:


clr_res = mirsig.clr(dream_df.values,128,16)


# In[ ]:


np.save('ADclr_res', clr_res)


# In[ ]:


genie_res = mirsig.parallel_genie(dream_df)


# In[ ]:


np.save('ADgenie', genie_res)


# In[ ]:


genie_no_dir = pd.DataFrame(np.maximum(genie_res.values, genie_res.values.T)) 


# In[ ]:


spearman_res = mirsig.spearman_df(dream_df)


# In[ ]:


pearson_res = mirsig.pearson_df(dream_df)


# In[ ]:


#mrnetb_res= mirsig.mrnetb_cuda()
pearson_res = pearson_res.abs()


# In[ ]:


spearman_res = spearman_res.abs()


# In[ ]:


kendall_res = mirsig.kendal_taub_cuda(dream_df)


# In[ ]:


borda_res = borda([mrnetb_res.values,
                   clr_res,
                   genie_res.values,
                   kendall_res.values,
                   spearman_res.values,
                   pearson_res.values],
                   pearson_res.shape[0], 6)
np.fill_diagonal(borda_res, 0)


# In[ ]:


np.save('ADdata', borda_res)


# In[ ]:


dS = pd.DataFrame (borda_res, columns= column_names, index= column_names )
edgeList = dS.stack().reset_index()
edgeList=edgeList[edgeList[0]>0.9]
edgeList[['level_0','level_1'] ].to_csv('AD_network_edgelist.tsv', sep='\t', header=False, index=False)

