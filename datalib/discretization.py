from datalib.utils import *
import numpy as np
import pandas as pd

def get_gap_names(cut_pt):
    """This function takes in a vector of cut points and returns a character vector of the names of the intervals formed by those cut points. 
    The intervals are named using the standard notation for intervals, with the lower and upper bounds separated by a comma and enclosed in parentheses. 
    The lower bound of the first interval is -infinity and the upper bound of the last interval is infinity.

    Parameters
    ----------
    cut_pt (numpy.array): The cut points considered for discretization.

    Returns
    -------
    aux (list) : Gaps.
    """
    
    aux = []
    aux.append(f"( -infty ,{cut_pt[0]}]")
  
    if len(cut_pt)>1:
        for i in range(len(cut_pt)-1):
            aux.append(f"({cut_pt[i]},{cut_pt[i+1]}]")
  
    aux.append(f"({cut_pt[-1]} , infty )")
    return(aux)

def new_name_dict(levels,cut_pt):
    """This function gets the names of the new categorical values and the cut points and returns a dictionary with the names as keys and the gaps as values.

    Parameters
    ----------
    levels (list): Names of the new discrete variables.
    cut_pt (numpy.array):Cut points
    Returns
    -------
    dict_names (dict): Dictionary with the new variable names as keys and the gaps as values.
    """
    dict_names={}
    level_names=get_gap_names(cut_pt)
    for i in range(len(levels)):
        dict_names[levels[i]]= f"{levels[i]}:{level_names[i]}"
    return(dict_names)

def discretize_generic (x, cut_pt):
    """This function gets a vector and a list of cut points and discretizes the vector by using them

    Parameters
    ----------
    x (list, np.array, pandas.DataFrame): The vector we want to discretize
    cut_pt (np.array): Array of cut points

    Returns
    -------
    pandas.Categorical: Discretized vector with categorical values
    """
    level = ["I"+str(i) for i in range(1,(len(cut_pt)+2))]# num_bins+1
    
    categorical = np.repeat (level[-1],len(x))
    for i in reversed(range(len(cut_pt))):
        categorical[x<=cut_pt[i]]=level[i]
    categorical = pd.Categorical(categorical,categories=level)
    return(categorical.rename_categories(new_name_dict(categorical.categories.tolist(),cut_pt)))

def discretizeEW(x, num_bins):
    """This function gets a vector and the number of bins that we want to create and discretizes it using the equal width algorithm.
    
    Equal width algorithm divides a given data set into a specified number of bins of equal width. 
    It is based on the idea of dividing the data into intervals or bins of equal size, and is useful for visualizing the distribution of continuous variables.

    Parameters
    ----------
    x (np.array): The vector we want to discretize
    num_bins (int): Number of bins we want to create

    Returns
    -------
    categorical (pandas.Categorical): Discretized vector with categorical values
    cut_pt (numpy.array):Cut points
    """
    if np.issubdtype(x.dtype, np.number):
        step = (max(x)-min(x))/(num_bins)
        cut_pt = np.repeat(min(x),num_bins-1)


        for i in range(len(cut_pt)):
            cut_pt[i:len(cut_pt)] = cut_pt[i:len(cut_pt)]+step

        categorical = discretize_generic (x, cut_pt)

        return(categorical, cut_pt)
    else: 
        return(x,[])

def discretizeEF(x, num_bins):  
    """This function gets a vector and the number of bins that we want to create and discretizes it using the equal frequency algorithm.
    
    Equal frequency algorithm divides a given data set into a specified number of bins with each bin containing the same number of data points. 
    It is based on the idea of dividing the data into intervals or bins such that each bin contains an equal number of data points, and is useful 
    for visualizing the distribution of discrete variables.

    Parameters
    ----------
    x (np.array): The vector we want to discretize
    num_bins (int): Number of bins we want to create

    Returns
    -------
    categorical (pandas.Categorical): Discretized vector with categorical values
    cut_pt (numpy.array):Cut points
    """
    if np.issubdtype(x.dtype, np.number):
        
        ordered = np.sort(x)    
        freq = round(len(x)/num_bins)
        cut_ind=range(freq,len(x),freq)
        cut_pt = ordered[cut_ind]

        if len(cut_pt)==1:   
            print("No hay partición")
            return(list(rep("I1",len(x)),np.nan))

        categorical = discretize_generic (x, cut_pt)
        return(categorical, cut_pt)
    
    else:
        return(x,[]) #Input data is not numeric

def discretize (data,num_bins, disc_alg ="EW"):
    
    """This function gets a data and the number of bins that we want to create and discretizes it using the equal width or the equal frecuency algorithm.

    Parameters
    ----------
    data (list, np.array, pandas.DataFrame): The data we want to discretize.
    num_bins (int): Number of bins we want to create
    disc_alg (str): The name of the algorithm we want to use. There are two options:
                    -EW: Equal width
                    -EF: Equal frequency
                    By default the equal width algorithm will be used

    Returns
    -------
    data (pd.DataFrame): Dataframe with discretized values.
    aux_p (list): List with the cut points.
    """
    data = np.transpose(data2df (data)) # transpose to loop columnwise
    
    if disc_alg == "EW":

        aux_p = []
        for i, row in enumerate(data.values):
            categorical, cut_pt = discretizeEW(row, num_bins)
            data.loc[data.index[i]]=categorical
            aux_p.append(cut_pt)
        return (np.transpose(data), aux_p)

    elif disc_alg == "EF":
        aux_p = []
        for i, row in enumerate(data.values):
            categorical, cut_pt = discretizeEF(row, num_bins)
            data.loc[data.index[i]]=categorical
            aux_p.append(cut_pt)
        return (np.transpose(data), aux_p)
    else:
        print("Either the algorithm you are trying to use or the name format is not recognized. Please select one of the following:\n -EW: Equal width\n -EF: Equal frequency") 
        