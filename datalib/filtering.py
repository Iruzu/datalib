from datalib.utils import *
import numpy as np
import pandas as pd

def filter_each_condition (data, var, condition, th_value, new_val=None):
    """
    This function applies a contition to a dataframe.
    
    Parameters
    ----------
    data (pandas.DataFrame): Data in which the condition will be applied.
    var (str): The variable that will be changed.
    condition (str): The logic operator. The options are <,>,<=,>=,== and !=.
    th_value (float): The threshold value for the condition.
    new_val : The new value the variables that fullfil the condition will take. By default it will be None and in that case the variables that fulfill the condition will simply be deleted.
    
    Returns
    -------
    data (pandas.DataFrame): The modified data.
    """
    
    ops = {"<": (lambda x,y: x<y), ">": (lambda x,y: x>y),"<=": (lambda x,y: x<=y), ">=": (lambda x,y: x>=y), "==": (lambda x,y: x==y), "!=": (lambda x,y: x!=y)}
    
    if new_val:
        data.loc[ops[condition](data[var], th_value), var] = new_val
        
    elif not new_val:
        data = data.loc[ops[condition](data[var], th_value)].reset_index(drop=True)
          
    return(data)

def filter_condition_list (data, condition_list):
    """
    This function takes data and filters/modifies it considering the conditions given in condition_list. 
    
    Each condition will have to be in the following format:
        -For filtering: "variable_to_be_modified logic_operator threshold" 
            ex.: "AUC != 0.8"
        -For modification: "variable_to_be_modified logic_operator threshold, new_value" 
            ex.: "entropy <= 2, 3"
    
    Parameters
    ----------
    data (list, matrix, numpy.array or pandas.DataFrame): Data to be modified.
    condition_list (list): A list with all the conditions to apply.
    
    Returns
    -------
    data_new (pandas.DataFrame): The modified data.
    """
    
    try:
        data_new = data2df(data).copy()
    except:
        return(None)
    
    condition_list = [condition.replace(",", "").split() for condition in condition_list]
    condition_list = [condition+[None] if len(condition)==3 else condition for condition in condition_list]
    
    for i, cond in enumerate(condition_list):
        if not len(cond)==4:
            print(f"There is something wrong with the format of condition number {i+1} so it won't be applyed.")
        else:
            var, condition, th_value, new_val = cond
            try:
                data_new[var]
                data_new = filter_each_condition (data_new, var, condition, float(th_value), new_val)
            except:
                print(f"La variable que se quiere editar con la condición {i+1} no sé encuentra en el dataset")
    return(data_new)
