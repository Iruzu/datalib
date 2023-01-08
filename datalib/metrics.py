from datalib.utils import *
import numpy as np
import pandas as pd

def entropy(x, normalize=False):
    """
    Calculates the entropy of the discrete vecor x.
    
    Entropy is a measure of the disorder or randomness of the data. It shows how much uncertainty or unpredictability is present in the data. 
    A data set with high entropy has a high degree of disorder, whereas a data set with low entropy is more predictable.
    
    Parameters
    ----------
    x (list, np. array or pandas.DataFrame): Vector of values.
    normalize (bool): If True the entropy will be normalized, if false it will not. By default the entropy will not be normalized.
    
    Returns
    -------
    float: Entropy of the vector.
    
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if isinstance(x, pd.Categorical):
        levels = x.categories
    else:
        levels = np.unique(x)
        
    prob = np.array([np.count_nonzero(x == level)/len(x) for level in levels])
    prob = np.delete(prob, np.argwhere(prob == 0))
    H = np.nansum(-prob*np.log(prob))
    if normalize and H!=0 and len(levels)!=1:
        return(H/np.log(len(levels)))
    else:
        return(H)

def variance(x, var_type = "population"):
    """This function calculates the variance of the continuous vector x. It will calculate the sample variance of the population variance depending the value of var_type.
    
    The variance is a measure of how spread out the data is. It is calculated by taking the average of the squared differences of each data point from the mean.
    
    The sample variance is calculated from a sample of the data, while the population variance is calculated from the entire population. 
    The sample variance uses n-1 as the denominator of the fraction to correct for bias in estimating the population variance, while the population variance uses n. 
    The sample variance has a higher variance than the population variance, meaning that it is more prone to fluctuations.
    
    Parameters
    ----------
    x (list, np. array or pandas.DataFrame): Vector of values.
    var_type (str): Specifyes the type of variance to be used. There are two options "population" and "sample". By default it will calculate the 
    population variance.

    Returns
    -------
    float: Variance of the vector.
    """
    if var_type == "population":
        n = len(x)
    elif var_type == "sample":
        n = len(x)-1
    else:
        print("The input is not between the options")
    
    return(np.sum((x - np.mean(x))**2) / n)

def AUC (df, return_TPR_FPR = False):
    """
    Calculates the area under the curve (AUC) of the df dataframe.
    For that it will first calculate the receiver operating characteristic (ROC) curve based on the data in the input dataframe, df.
    
    The AUC is a measure of the classifier's performance. A classifier that is 100% accurate will have an AUC of 1.0, whereas a classifier that is no better than 
    random guessing will have an AUC of 0.5.
    
    Parameters
    ----------
    df (pandas.DataFrame): The dataframe containing the probabilities and labels of the classifier to be evaluated.
    
    return_TPR_FPR (bool, optional): Determines whether the function returns only the AUC or also the true positive rate (TPR) and false positive rate (FPR) arrays 
        used to calculate the AUC. Defaults to False.
    
    Returns
    -------
    tuple: If return_TPR_FPR is True, returns a tuple containing the AUC and the TPR and FPR arrays. If return_TPR_FPR is False, returns only the AUC.
    
    Notes
    -----
    - The input dataframe, df, should contain a single column of boolean values representing the true labels and a single column of continuous values between 0 and 1 
       representing the predicted probabilities. If this conditions are not fullfilled the function will return None.
    - If the continuous values in df are not probabilities between 0 and 1 the function will also return None.
    """
    
    labels = df.select_dtypes(include=[bool])
    val = df.select_dtypes(exclude=[bool])
    [val_index] = val.columns.tolist()
    [lab_index] = labels.columns.tolist()
    
    if len(labels.columns)>1:
        print("There is more than one boolean variable.")
        return(None)
    if len(val.columns)>1:
        print("There is more than one predictions variable.")
        return(None)
    if float(val.max())>1 or float(val.min())<0:
        print("The predictions should be probabilityes between 0 and 1. This condition is not fullfilled. please check the input.")
        return(None)
    
    
    df = df.sort_values(by= [val_index])
    df["aux"]= True
    
    TPR_lis= []
    FPR_lis= []
    
    for v_corte in df[val_index]:

        df.loc[df[val_index] <v_corte, "aux"] = False
        
        TP = sum([1 if (df["aux"].iloc[i] and df[lab_index].iloc[i]) else 0 for i in range (len(df))])
        FP = sum([1 if (df["aux"].iloc[i] and not df[lab_index].iloc[i]) else 0 for i in range (len(df))])
        TN = sum([1 if (not df["aux"].iloc[i] and not df[lab_index].iloc[i]) else 0 for i in range (len(df))])
        FN = sum([1 if (not df["aux"].iloc[i] and df[lab_index].iloc[i]) else 0 for i in range (len(df))])

        TPR = TP/(TP + FN)
        FPR = FP/(FP + TN)

        TPR_lis.append(TPR)
        FPR_lis.append(FPR)

    aux = FPR_lis[1:]+[0]
    steps = abs(np.subtract(FPR_lis,aux))
    
    AUC = sum(TPR_lis*steps)
    
    if return_TPR_FPR:
        return(AUC,TPR_lis,FPR_lis)
    else:
        return(AUC)

def calc_metrics (data):
    """
    This function calculates the metrics for the input data acording to its type; entropy if the variable is discrete, variance if it is continuous and AUC if the data is continuous 
    with a boolean variable.
    
    Parameters
    ----------
    data (list, numpy.array or pandas.DataFrame): The data whose metrics are going to be calculated.
    
    Returns
    -------
    (int or pd.Series): The calculated metric according to the type of data.
    """
    
    try:
        data_new = data2df(data).copy()
    except:
        return(None)
    
    if any(data_new.dtypes == 'bool') and is_cont(data_new.select_dtypes(exclude=[bool])): 
        return(AUC(data_new))
                                            
    elif is_cont(data_new):                   
        v = data_new.apply(variance, axis=0)
        v.name = "Variance"
        return(v)
    
    elif not is_cont(data_new) and not any(data_new.dtypes == 'float64'): 
        e = data_new.apply(entropy, axis=0)
        e.name = "Entropy"
        return(e)
        
    else: # igual los datos son mixtos
        met = []
        data_new = np.transpose(data_new)
        for row in data_new.values:
            if row.dtype == 'float64':
                met.append(variance(row))
            else:
                met.append(entropy(row))
        met = pd.Series(met)
        met.name = "Variance & Entropy"
        return(met)
