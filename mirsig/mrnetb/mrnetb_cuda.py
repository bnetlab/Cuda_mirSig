
# coding: utf-8

# In[1]:



# import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np 
import pandas as pd
from pycuda.compiler import SourceModule
import math

def mrnetb_cuda(df):


    
    start = cuda.Event()
    end = cuda.Event()


    # In[2]:


    # sampleSize = -1
    # numVars = -1# -1 for all


    # df = pd.read_csv('data/TCGA-PRAD-tumor.csv', sep=',', index_col=0)

    # Simulated Gene Expression Data check :\
    # https://www.bioconductor.org/packages/release/bioc/manuals/minet/man/minet.pdf
    # df = pd.read_csv('data/syn_data.csv', sep=',', index_col=0)
    # print (df)
    # # remove zeros
    # df = df.loc[:,(df != 0).any(axis=0)]

    # sampleSize = (sampleSize) if (sampleSize!= -1 ) else df.shape[0]
    # numVars = (numVars) if (numVars!= -1 ) else df.shape[1]

    # df = df.iloc[0:sampleSize,0:numVars]


    # In[3]:
    sampleSize = df.shape[0]
    numVars = df.shape[1]

    # clac mi 
    cors = df.corr(method='pearson')**2
    np.fill_diagonal(cors.values,0)
    vals = cors.values 
    mi = -0.5*np.log(1-vals)
    sums = mi.sum(axis=0)


    # In[4]:



    mod = SourceModule("""
    #define NUMTHREADS 16
    #define THREADWORK 1
    #define ELEMINATED 10000

    /* curVar is current variablle which we can find out by tx,ty,bx, by 
    # that is 
    # idx = blockIdx.x*blockDim.x+threadIdx.x
    # idy = blockIdx.y*blockDim.y+threadIdx.y
    # totx = blockDim.x*gridDim.y
    # curVar = idx*(blockDim.)
    # get all the mi that is globale 
    # n is size of mi 
    # rels is relevancy for current var with size n *n 
    # reds is reudundencies for current var with size n *n size they are global
    */

    __global__ void gpuMrnetb(float *mi, float *rels, float *reds, float *sums, float *res, size_t size)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int totx = gridDim.x * blockDim.x;
    int curVar = idx * totx + idy;
    int temp;

    if(curVar > size)
        return;
    int k = 0 ;
    int  i, worst, best;

    for (i = 0; i < size; i++)
    {
        rels[curVar * size + i] = mi[curVar * size + i];
        reds[curVar * size + i] = sums[i] - mi[i * size + curVar];

        k++;
    }

    worst = 0;

    //select worst
    for (i = 1; i < size; i++)
    {
        if ((rels[curVar * size + i] - reds[curVar * size + i] / k) < (rels[curVar * size + worst] - reds[curVar * size + worst] / k))
        {
            worst = i;
        }
    }

    best = worst;

    //backward elimination
    while ((k > 1) && ((rels[curVar * size + worst] - reds[curVar * size + worst] / k) < 0))
    {

        //eliminate worst
        rels[curVar * size + worst] = ELEMINATED;

        k--;

        for (i = 0; i < size; i++)
        {
            reds[curVar * size + i] -= mi[i * size + worst];
        }
        worst = 0;

        for (i = 1; i < size; i++)
        {

            if ((rels[curVar * size + i] - reds[curVar * size + i] / k) < (rels[curVar * size + worst] - reds[curVar * size + worst] / k))
                worst = i;
        }
    }

    //sequential replacement
    for (i = 0; i < size; i++)
    {

        if (rels[curVar * size + i] == ELEMINATED)
        {
            if ((mi[curVar * size + i] - reds[curVar * size + i] / k) > (mi[curVar * size + best] - reds[curVar * size + best] / k))
                best = i;
        }
    }

    int ok = 1;
    while (ok)
    {
        rels[curVar * size + best] = mi[curVar * size + best];
        rels[curVar * size + worst] = ELEMINATED;

        for (i = 0; i < size; i++)
            reds[curVar * size + i] += mi[i * size + best] - mi[i * size + worst];

        temp = best;
        best = worst;
        worst = temp;

        ok = 0;
        for (i = 0; i < size; i++)
        {
            if (rels[curVar * size + i] != ELEMINATED)
            {
                if ((rels[curVar * size + i] - reds[curVar * size + i] / k) < (rels[curVar * size + worst] - reds[curVar * size + worst] / k))
                {
                    worst = i;
                    ok = 1;
                }
            }
            else
            {
                if ((mi[curVar * size + i] - reds[curVar * size + i] / k) > (mi[curVar * size + best] - reds[curVar * size + best] / k))
                {
                    best = i;
                    ok = 1;
                }
            }
        }
    }

    for (i = 0; i < size; i++)
    {
        if (rels[curVar * size + i] == ELEMINATED)
            res[curVar * size + i] = 0;
        else
            res[curVar * size + i] = rels[curVar * size + i] - reds[curVar * size + i] / k;
    }
    }
    """)


    # In[5]:


    ## complie 
    func = mod.get_function("gpuMrnetb")


    # In[6]:


    h_results = np.empty(numVars*numVars)
    h_results = h_results.astype(np.float32)
    h_reds = np.empty(numVars*numVars)
    h_reds = h_results.astype(np.float32)
    h_rels = np.empty(numVars*numVars)
    h_rels = h_rels.astype(np.float32)
    h_sums = sums.astype(np.float32)
    h_mi = np.empty(numVars*numVars)
    h_mi = mi.astype(np.float32)


    # # In[7]:


    # sums


    # # In[8]:


    d_results = cuda.mem_alloc(h_results.nbytes)
    d_reds = cuda.mem_alloc(h_reds.nbytes)
    d_rels = cuda.mem_alloc(h_rels.nbytes)
    d_sums = cuda.mem_alloc(h_sums.nbytes)
    d_mi = cuda.mem_alloc(h_mi.nbytes)


    cuda.memcpy_htod(d_mi, h_mi)
    cuda.memcpy_htod(d_sums, h_sums)


    # In[9]:


    start.record()
    blocksize = 16 
    gridSize = math.floor(numVars/blocksize)+1
    func(d_mi, d_rels, d_reds, d_sums, d_results, np.int32(numVars), block=(16,16,1), grid =(gridSize,gridSize,1))

    end.record()


    # In[10]:


    end.synchronize()
    secs = start.time_till(end)*1e-3
    print ("Mrnetb for :",numVars)
    print ("%fs sec" % (secs))


    # In[11]:


    cuda.memcpy_dtoh(h_results, d_results)


    # # In[12]:


    res = np.reshape(h_results,(numVars,numVars))
    res = np.maximum(res, res.T)
    # normalization
    res = res/res.max()


    # In[13]:


    result = pd.DataFrame(data=res)
    return result

