import numpy as np
import pandas as pd

def data2df(data):
    """This function gets a vector or matrix and returns it in pandas Dataframe format.
    If the input data is already in pandas Dataframe format the output will be the same as the input.

    Parameters
    ----------
    data (list, numpy.array or pandas.DataFrame): The data whose format is going to be changed.

    Returns
    -------
    pd.DataFrame : Input data in pandas Dataframe format
    """
    
    if not isinstance(data, pd.DataFrame):
        if len(np.array(data).shape)==1:
            return(pd.DataFrame(data,columns=["Var 1"]))
        elif len(np.array(data).shape)==2:
            names= [f"Var {i+1}" for i in range(len(data[0]))]
            return(pd.DataFrame(data, columns=names))
        else:
            print("Something is wrong with the input. Check that it is in one of the following formats:\n -Vector (List or numpy array)\n -Matrix (List or numpy array)\n -Pandas DataFrame")
    else:
        return(data)
    
def is_num(data):
    """This function checks if the input data is numeric. Returns True if it is and False if it is not.

    Parameters
    ----------
    data (pandas.DataFrame): The data whose type is going to be checked.

    Returns
    -------
    bool : True if input is numeric, False if not.
    """
    if all(data.dtypes == 'float64') or all(data.dtypes == 'int64'):
        return(True)
    else:
        return(False)

def is_cont(data):
    """This function checks if the input data is continuous. Returns True if it is and False if it is discrete.

    Parameters
    ----------
    data (pandas.DataFrame): The data whose type is going to be checked.

    Returns
    -------
    bool : True if input is continuous, False if it is discrete.
    """
    if all(data.dtypes == 'float64'):
        return(True)
    elif isinstance(data, pd.Categorical) or isinstance(data, str) or all(data.dtypes == 'int64'):
        return(False)