
# coding: utf-8

# The Tau-b statistic, unlike Tau-a, makes adjustments for ties.[5] Values of Tau-b range from −1 (100% negative association, or perfect inversion) to +1 (100% positive association, or perfect agreement). A value of zero indicates the absence of association.
# 
# The Kendall Tau-b coefficient is defined as:
# 
# $$\tau _{B}={\frac {n_{c}-n_{d}}{\sqrt {(n_{0}-n_{1})(n_{0}-n_{2})}}}$$
# 
# where
# $$
# {\displaystyle {\begin{aligned}n_{0}&=n(n-1)/2\\n_{1}&=\sum _{i}t_{i}(t_{i}-1)/2\\n_{2}&=\sum _{j}u_{j}(u_{j}-1)/2\\n_{c}&={\text{Number of concordant pairs}}\\n_{d}&={\text{Number of discordant pairs}}\\t_{i}&={\text{Number of tied values in the }}i^{\text{th}}{\text{ group of ties for the first quantity}}\\u_{j}&={\text{Number of tied values in the }}j^{\text{th}}{\text{ group of ties for the second quantity}}\end{aligned}}}$$
# 
# 
# 
# https://github.com/nullsatz/gputools/tree/master/src

# In[1]:


# import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np 
import pandas as pd
import math
from pycuda.compiler import SourceModule




# In[2]:

def _kendal_taub_cuda(df):
## Cuda kernel

        start = cuda.Event()
        end = cuda.Event()

        mod = SourceModule("""

        typedef struct {
        float t; //join ties
        float u; //x ties
        float v; //y ties
        float c; //concordant
        }pars;

        #define NUMTHREADS 16
        #define THREADWORK 1
        // na and nb are number of cols
        // we use a = b ; 
        __global__ void gpuKendallDB(float * a, int na, 
                float * b, int nb, int sampleSize, float * results) 
        {
                size_t 
                        i, j, tests, 
                        tx = threadIdx.x, ty = threadIdx.y, 
                        bx = blockIdx.x, by = blockIdx.y,
                        rowa = bx * sampleSize, rowb = by * sampleSize;
                        //printf("%d blockIdx.x , %d blockIdx.y \\n", blockIdx.x, blockIdx.y);
                float 
                        discordant,result, concordant, u,v,t , n = 0.f, 
                        numer, denom;

                __shared__ pars threadSums[NUMTHREADS*NUMTHREADS];

                for(i = tx; i < sampleSize; i += NUMTHREADS) {
                        for(j = i+1+ty; j < sampleSize; j += NUMTHREADS) {
                                tests = ((a[rowa+j] >  a[rowa+i]) && (b[rowb+j] >  b[rowb+i]))
                                        + ((a[rowa+j] <  a[rowa+i]) && (b[rowb+j] <  b[rowb+i])) ;
                                // + ((a[rowa+j] == a[rowa+i]) && (b[rowb+j] == b[rowb+i])); 
                                concordant = concordant + (float)tests;
                                u += (a[rowa+j] == a[rowa+i]);
                                v += (b[rowb+j] == b[rowb+i]);
                                t += ((a[rowa+j] == a[rowa+i]) && (b[rowb+j] == b[rowb+i]));
                        }
                }
                threadSums[tx*NUMTHREADS+ty].c = concordant;
                threadSums[tx*NUMTHREADS+ty].v = v;
                threadSums[tx*NUMTHREADS+ty].u = u;
                threadSums[tx*NUMTHREADS+ty].t = t; 

                __syncthreads();
                for(i = NUMTHREADS >> 1; i > 0; i >>= 1) {
                        if(ty < i){ 
                                threadSums[tx*NUMTHREADS+ty].c += threadSums[tx*NUMTHREADS+ty+i].c;
                                threadSums[tx*NUMTHREADS+ty].v += threadSums[tx*NUMTHREADS+ty+i].v;
                                threadSums[tx*NUMTHREADS+ty].u += threadSums[tx*NUMTHREADS+ty+i].u;
                                threadSums[tx*NUMTHREADS+ty].t += threadSums[tx*NUMTHREADS+ty+i].t;
                                }
                        __syncthreads();
                }
        for(i = NUMTHREADS >> 1; i > 0; i >>= 1) {
        if((tx < i) && (ty == 0)){
        threadSums[tx*NUMTHREADS].c += threadSums[(tx+i)*NUMTHREADS].c;
        threadSums[tx*NUMTHREADS].u += threadSums[(tx+i)*NUMTHREADS].u;
        threadSums[tx*NUMTHREADS].v += threadSums[(tx+i)*NUMTHREADS].v;
        threadSums[tx*NUMTHREADS].t += threadSums[(tx+i)*NUMTHREADS].t;
        }
        __syncthreads();
        }
                
                if((tx == 0) && (ty == 0)) {
                        concordant = threadSums[0].c;
                        u = threadSums[0].u;
                        v = threadSums[0].v;
                        t = threadSums[0].t;
                        denom = sampleSize;
                        //printf(blockIdx.y);
                        //printf("u = %0.2f , v = %0.2f , t = %0.2f , bx = %d , by = %d \\n", u,v,t, bx, blockIdx.y);
                        
                        // remember tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
                        //# Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
                        //= con + dis + xtie + ytie - ntie

                        //???

                        // denom = (denom * (denom - 1.f)) / 2.f; discordant = denom - concordant - u - v + t;
                        // denom = sqrt((discordant + concordant+v)*(discordant + concordant + u));
                        // numer = concordant - discordant;
                        
                        // Wikipedia algorithm 
                        n = (sampleSize * (sampleSize - 1.f))/2.f; 
                        discordant = n - concordant - u - v + t; 
                        denom = sqrtf((n-u)*(n-v));
                        numer = concordant - discordant; 

                        // results (rounded to 6 decimal places)
                        
                        //results[ blockIdx.y*na+bx] = floorf((numer)/(denom)*100000)/100000;
                        result = (numer)/(denom);
                        
                        results[ blockIdx.y*na+bx] = (result>1) ? 1.f : result;
                        //printf("%0.2f",results[ blockIdx.y*na+bx]);
                }
        }
        """)


        # Data has 119 rows with all zero values

        # In[3]:


        # sampleSize = -1
        # numVars = -1# -1 for all
        # Cpu = False # run on cpu too

        # df = pd.read_csv('data/TCGA-PRAD-tumor.csv', sep=',', index_col=0)
        # # remove zeros


        # sampleSize = (sampleSize) if (sampleSize!= -1 ) else df.shape[0]
        # numVars = (numVars) if (numVars!= -1 ) else df.shape[1]

        # df_num = df.values[0:sampleSize,0:numVars] # get np array
        # df_num.shape
        # print(sampleSize,numVars)
        df_num = df.values
        sampleSize = df_num.shape[0]
        numVars = df_num.shape[1]
        h_a = df_num.T #Transpose for better performance in kernel


        # In[4]:


        #  pd.options.display.max_rows = 999
        # ((df!=0).any(axis=0))[((df!=0).any(axis=0))==False ]


        # In[5]:


        h_a = np.reshape(h_a,numVars*sampleSize) #Reshape into 1D
        h_a = h_a.astype(np.float32) 


        # In[6]:


        h_results = np.empty(numVars*numVars)
        h_results = h_results.astype(np.float32)

        d_results = cuda.mem_alloc(h_results.nbytes)
        a_gpu = cuda.mem_alloc(h_a.nbytes)
        cuda.memcpy_htod(a_gpu, h_a)

        #compile
        func = mod.get_function("gpuKendallDB")



        # In[7]:


        start.record()

        func(a_gpu,np.int32(numVars) ,a_gpu,np.int32(numVars),np.int32(sampleSize),
        d_results,block=(16,16,1),grid=(numVars,numVars,1))

        end.record()


        # In[8]:


        end.synchronize()
        secs = start.time_till(end)*1e-3
        print ("Kendall for :",numVars)
        print ("%fs sec" % (secs))


        # In[9]:


        cuda.memcpy_dtoh(h_results, d_results)
        np.round(h_results,decimals=6) 

        #there is small(<1.0e+-9) difference between calculated values from gpu and cpu
        # due to device arthimatics


        # In[10]:


        results = pd.DataFrame(data=np.reshape(h_results,(numVars,numVars)))

        return results
        # # Tests Results 

def kendal_taub_cuda(df):
        
        result = _kendal_taub_cuda(df)
        result = result.abs()
        np.fill_diagonal(result.values, 0)
        return result

        # In[11]:


        # if Cpu :


        # # In[ ]:


        # df = pd.DataFrame(data=df_num);


        # # In[ ]:


        # get_ipython().run_cell_magic('time', '', "h_test = df.corr(method='kendall')")


        # # In[ ]:


        # diff =(h_test-results).values
        # np.testing.assert_equal(diff[diff > 0.0001],np.empty([]))


        # # # Benchmarks 

        # # samplesize: 498 numvars: 1024  
        # #     GPU times: user 855 µs, sys: 1.05 ms, total: 1.91 ms
        # #     CPU times: user 4min 15s, sys: 74.4 ms, total: 4min 15s

        # # # Plots 

        # # In[ ]:


        # get_ipython().run_line_magic('matplotlib', 'inline')
        # from pandas.tools.plotting import scatter_matrix
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import mpld3
        # mpld3.enable_notebook()
        # plt.rcParams['figure.figsize'] = 30, 20 


        # # In[ ]:



        # results.iloc[15000].diff().hist(bins=100)


# In[ ]:



# plt.pcolor(results)
# plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
# plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
# mpld3.show()

