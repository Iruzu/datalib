import numpy as np
import matplotlib.pyplot as plt

def plot_ROC (TPR, FPR, AUC=None):
    """
    This function plots a receiver operating characteristic (ROC) curve and optionally displays the area under the curve (AUC).
    
    Parameters
    ----------
    TPR (list or array): List or array of true positive rates at different thresholds.
    FPR (list or array): List or array of false positive rates at different thresholds. Must be the same length as TPR.
    AUC (float, optional): AUC value to be displayed on the plot.
    
    Returns
    -------
    None    
    """
    
    plt.plot(TPR,FPR,color = "slateblue", zorder=1)
    plt.scatter(TPR,FPR, s=80, marker = "*", color = "darkorange", zorder=2)
    plt.title("ROC curve")
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    if AUC:
        plt.fill_between(TPR,FPR,color = "slateblue", step="pre", alpha=0.4, label="AUC = "+str(AUC))
        plt.legend()
    plt.show()

def plot_CM (corr_matrix):
    """
    This function plots a correlation matrix for a given dataframe.
    
    Parameters
    ----------
    corr_matrix (pandas dataframe): Dataframe containing the correlations between pairs of variables.
    
    Returns
    -------
    None    
    """
    var_names = corr_matrix.columns.tolist()
    
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 4))
    
    mat = ax.matshow(corr_matrix, cmap="inferno")    
    fig.colorbar(mat)
    
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45)
    ax.tick_params(axis="x",bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_yticks(range(len(var_names)))
    ax.set_yticklabels(var_names)
    
    plt.title('Correlation Matrix')
    plt.show()

def plot_entropy (x):
    """
    Plots a bar chart of entropy values.
    
    Parameters
    ----------
    x (list or array): List or array of entropy values to be plotted.
    
    Returns
    -------
    None 
    """
    names= [f"Var {i+1}" for i in range(len(x))]
    plt.bar(names,x,color = "slateblue")
    plt.title("Entropy")
    plt.ylabel('Value')
    plt.show()