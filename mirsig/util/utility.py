from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support ,f1_score
from sklearn.metrics import average_precision_score , auc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import re

def get_data (file ,sampleSize = -1 , numVars = -1, *args, **kwargs ):
    """ 
    Loads gene_exprssion data file and remove columns in which all values are zero.
    
    Parameters
    ----------
    df : array, shape = [n]
        x coordinates.
    file : array, shape = [n]
        y coordinates.
    samlpeSize : int
        -1 for all 
    
    numVars : int
        -1 for all 
    Returns
    -------
    df : pandas array, geneNames: numpy array of columns names, sampleSize, numVars

    """
    df = pd.read_csv(file, *args, **kwargs)
    df = df.loc[:,(df != 0).any(axis=0)]

    sampleSize = (sampleSize) if (sampleSize!= -1 ) else df.shape[0]
    numVars = (numVars) if (numVars!= -1 ) else df.shape[1]

    df = df.iloc[0:sampleSize,0:numVars]
    geneNames = df.columns
    return df, geneNames , sampleSize, numVars


def get_edge_list (mat_df, sort = True):
    edgelist = mat_df.copy()
    edgelist.values[[np.arange(len(edgelist))]*2] = np.nan
    edgelist = edgelist.stack().reset_index()

    edgelist.columns = ["G1","G2","w"]
    if sort:
        edgelist = edgelist.sort_values('w' ,axis = 0, ascending=False)

    return edgelist



#create NXX matrix from edge list, works for dREAM data
def get_mat_from_edge_list(edge_df, numVars):
    edge_df = edge_df[edge_df[2] != 0 ]
    edge_df = pd.crosstab(edge_df[0], edge_df[1])
    edge_df.columns = edge_df.columns.map(lambda x : int(re.sub('G','',x)))
    edge_df.index = edge_df.index.map(lambda x : int(re.sub('G','',x)))
    cols = [x for x in range(1, edge_df.columns.max()+1)] 
    idx = [x for x in range(1, edge_df.index.max()+1)]
    edge_df = edge_df.reindex(index=idx, columns=cols, fill_value=0)
    k = edge_df.shape[0]
    edge_df_dir = edge_df.copy()
    edge_df = pd.DataFrame(np.pad(edge_df.values, ((0,edge_df.shape[1]-edge_df.shape[0]),(0,0)), mode= 'constant',  constant_values=0))
    edge_df[:] = np.maximum(edge_df.values, edge_df.values.T)
    edge_df = edge_df[:k] # Idk
    # edge_df = edge_df[edge_df[2] != 0 ]
    # mat_df = pd.crosstab(edge_df[0], edge_df[1])
    # idx = mat_df.columns.union(mat_df.index)
    # idxAll = ['G'+str(x) for x in range(1,numVars)]
    # idx = idx.union(idxAll)
    # mat_df = mat_df.reindex(index=mat_df.index, columns=idx, fill_value=0)
    # #fix index and columnt ordering 
    # mat_df.columns = mat_df.columns.map(lambda x : re.sub('G','',x) )
    # mat_df = mat_df.reindex(sorted(mat_df.columns, key=float), axis=1)

    # mat_df.index = mat_df.index.map(lambda x : re.sub('G','',x) )
    # mat_df = mat_df.reindex(sorted(mat_df.index, key=float), axis=0)
    # mat_df[:]= np.maximum(mat_df.values,mat_df.values.T)
    return (edge_df, edge_df_dir)


def calc_auprc(y_true_df, y_pred_df):
    
    y_true  = y_true_df.values
    y_true  = y_true.flatten()
    y_pred = y_pred_df.values.flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aucpr = auc(recall, precision)
    return precision, recall, thresholds ,aucpr 

def get_f1_score(y_true_df, y_pred_df, random = False, random_num = 100):
    """ 
    Calculated F1_score
    
    Parameters
    ----------
    y_true_df : pandas array
        matrix of gold standtard edge weights nXn
    y_pred_df : pandas array
        matrix of predicted edge weights nXn
    random: Boolean
        random sampling the threshholds in order to improve performance
    random_num : Number
        number of random samples
    
    Returns
    -------
    df : pandas dataframe, 
        with f1_score and associated threshold with columns 'f1_score', 'thresholds'
    

    """
    
    precision, recall, thresholds, auprc = calc_auprc(y_true_df, y_pred_df)
    y_pred = y_pred_df.values.flatten()
    if random:
        thresholds = np.random.choice(thresholds,random_num)   
    f_scores = [ [(f1_score((y_true_df.values.flatten()==1), (y_pred>threshold), average= 'binary')), threshold] for threshold in thresholds]
    return pd.DataFrame(f_scores, columns=['f_score', 'threshold'])

    # get f score for each thresholds
    
    
    return precision, recall, thresholds  

def plot_auprc(recall, precision, auprc):

    plt.figure(figsize=(15,10))
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUPRC={0:0.5f}'.format(
          auprc))