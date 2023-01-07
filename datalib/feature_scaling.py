from datalib.utils import *
import numpy as np
import pandas as pd

def normalize(x):
    """
    This function normalizes a vector such that its values range from 0 to 1.
    
    Parameters
    ----------
    x (numpy.array or pandas.DataFrame): The vector to be normalized.
    
    Returns
    -------
    x_norm (numpy.array or pandas.Series): The normalized vector.
    """
    x_norm = (x-min(x))/ (max(x)-min(x))
    return x_norm

def standarize(x):
    """
    This function calculates the metrics for the input data acording to its type; entropy if the variable is discrete, variance if it is continuous and AUC if the data is continuous 
    with a boolean variable.
    
    Parameters
    ----------
    x (numpy.array or pandas.DataFrame): The vector to be standarized.
    
    Returns
    -------
    x_stand (numpy.array or pandas.Series): Thestandarized vector.
    """
    x_stand = (x-np.mean(x))/np.std(x)
    return x_stand

def feature_scaling (data, operation="normalize"):
    """
    This function calculates the metrics for the input data acording to its type; entropy if the variable is discrete, variance if it is continuous and AUC if the data is continuous 
    with a boolean variable.
    
    Parameters
    ----------
    data (list, matrix, numpy.array or pandas.DataFrame): Data to be normalized or standarized.
    operation (str): The type of feature scaling to be performed. The options are "normalize" or "satandarize". By default the function will normalize the data.
    
    Returns
    -------
    data_new (pandas.DataFrame): Normalized or standarized data.
    """
    
    try:
        data_new = data2df(data).copy()
    except:
        return(None)
    
    if is_num(data_new):
        if operation =="normalize":
            data_new = data_new.apply(normalize, axis=1)
        elif operation =="satandarize":
            data_new = data_new.apply(standarize, axis=1)
        return(data_new)
    else:
        print("The input data is not numeric")
