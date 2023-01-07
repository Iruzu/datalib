from datalib.utils import *
import numpy as np
import pandas as pd

def pearsons_correlation (x,y):   
    """This function calculates the Pearson's correlation coefficient between the vectors x and y. It will return a value between 0 and 1.
    The returned value can be interpreted as follows:
    
    -1 : strong negative correlation
     0 : no correlation
     1 : strong positive correlation.
    
    Parameters
    ----------
        x (array-like): The first vector.
        y (array-like): The second vector.
        
    Returns
    -------
        r (float): Pearson's correlation coefficient between x and y.
    """
        
    r = sum((x - np.mean(x)) * (y - np.mean(y))) / np.sqrt(sum((x - np.mean(x))**2) * sum((y - np.mean(y))**2))
    return(r)

def entropy_from_prob (prob, normalize=False):
    """
    Calculates the entropy from the probabilities.
    
    Entropy is a measure of the disorder or randomness of the data. It shows how much uncertainty or unpredictability is present in the data. 
    A data set with high entropy has a high degree of disorder, whereas a data set with low entropy is more predictable.
    
    Parameters
    ----------
    prob (np. array): Vector of probabilities.
    normalize (bool): If True the entropy will be normalized, if false it will not. By default the entropy will not be normalized.
    
    Returns
    -------
    float: Entropy of the vector.
    
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    n_levels = len(prob)
    prob = np.delete(prob, np.argwhere(prob == 0))
    H = np.nansum(-prob*np.log(prob)) # nansum just in cas
    if normalize and H!=0 and n_levels>1:
        return(H/np.log(n_levels))
    else:
        return(H)

def mutual_info(x, y):
    """
    Calculate the mutual information between two vectors.
    
    Mutual information is a measure of the dependence between two variables. It quantifies the amount of information that one variable contains about the other. 
    In other words, it measures how much knowing the value of one variable reduces uncertainty about the value of the other variable. 
    Mutual information is a non-negative value, and the more correlated the variables are, the higher the mutual information will be.
    
    Parameters
    ----------
    x (np. array): The first vector.
    y (np. array): The second vector.
    
    Returns
    -------
    float: Mutual information between x and y.
    """
    
    x_uniq = np.unique(x)
    y_uniq = np.unique(y)
    
    px = np.array([len(x.loc[x==x_val])/len(x) for x_val in x_uniq])
    Hx = entropy_from_prob(px)
    
    py = np.array([len(y.loc[y==y_val])/len(y) for y_val in y_uniq])
    Hy = entropy_from_prob(py)
    
    pxy=[]
    for y_val in y_uniq:
        ind = y.loc[y==y_val].index.tolist()
        x_aux = x.loc[ind] # x in whitch it ir y
        pxy.append([len(x_aux.loc[x_aux == x_val])/len(x) for x_val in x_uniq])
        
    pxy = np.transpose(pxy)
    Hxy = sum([entropy_from_prob(pi) for pi in pxy])
    return Hx+Hy- Hxy

def correlation(data):
    """
     This function calculates Pearson's correlation between pairs of columns if the data is continuous and mutual information if it is categorical.
     
     Parameters
     ----------
     data (pandas.dataframe): Data for which correlations will be calculated.
     
     Returns
     -------
     pandas.dataframe: Dataframe containing the correlations between pairs of columns. The rows and columns are labeled with the variable names.
     """    
    
    try:
        data = data2df(data)
    except:
        return(None)
    
    N = len(data.columns)
    names = data.columns.tolist()
    mi = np.zeros((N,N))
    
    if is_num(data):
        for i in range(N):
            for j in range(i,N):
                mi[i,j] = pearsons_correlation(data[names[i]],data[names[j]])
                mi[j,i] = pearsons_correlation(data[names[i]],data[names[j]])
        return(pd.DataFrame(mi,columns=names, index =names))
    
    elif not is_num(data):
        for i in range(N):
            for j in range(i,N):
                mi[i,j] = mutual_info(data[names[i]],data[names[j]])
                mi[j,i] = mutual_info(data[names[i]],data[names[j]])
        return(pd.DataFrame(mi,columns=names, index =names))