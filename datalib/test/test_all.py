from datalib import discretization
from datalib import metrics
from datalib import correlation
from datalib import filtering
from datalib import feature_scaling
from datalib import plotting
from datalib import utils
import numpy as np
import pandas as pd

permutation_matrix=[[1,5,2,4,3],[1,5,4,3,2],[2,5,1,3,4],[1,4,5,3,2],[3,5,4,1,2],[1,2,3,4,5],[5,4,3,2,1],[2,3,5,4,1]]
permutation_df = pd.DataFrame(permutation_matrix,columns=["Var1","Var2","Var3","Var4","Var5"])
single_permutation = [1,5,2,4,3]

categorical_matrix=[["a","a","c","c","a"],["a","a","b","c","a"],["b","b","c","c","a"],["a","b","c","a","a"],["a","a","a","c","a"],["b","b","c","c","c"]]
categorical_df= pd.DataFrame(categorical_matrix,columns=["Var1","Var2","Var3","Var4","Var5"])
single_categorical = ["a","a","c","c","a"]

cont_matrix=[[1.2,5.4,2.1,4.5,3.3],[1,5.3,4.9,3,2.2],[2,5.4,1.4,3.1,4.9],[1,4.1,5,3.4,2],[3,5.2,4,1.3,2],[1,2.4,3,4.5,5],[5,4.3,3,2.1,1],[2,3.7,5,4.3,1]]
cont_df = pd.DataFrame(cont_matrix,columns=["Var1","Var2","Var3","Var4","Var5"])
clasif_res = pd.DataFrame({"y":[0.1,0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], "lab":[False, False, True, False, True, False, False, True, True, True]})

def test_all():
    """Executes the tests to verify the installation has ben successfull"""
    
    # discretization
    discretization.discretize([11.5,10.2,1.2,0.5,5.3,20.5,8.4],4)   
    discretization.discretize(permutation_matrix,2, "EF")
    
    # metrics
    metrics.calc_metrics(clasif_res)   
    metrics.calc_metrics(categorical_df)
    metrics.calc_metrics(permutation_df)
    metrics.calc_metrics(cont_matrix)
    
    # correlation
    correlation(categorical_df)
    correlation(permutation_df)
    correlation(cont_df)
    
    # filtering
    test = metrics.calc_metrics(permutation_df)
    condition_list=["entropy <= 2, 3", "Var3 != 2",  "entorpy != 2"]
    filtering.filter_condition_list (test, condition_list)
    
    #feature_scaling
    feature_scaling(permutation_df,"normalize")
    feature_scaling(permutation_df,"satandarize")
    
    #plotting
    TPR=[1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.6, 0.4, 0.2]
    FPR=[1.0, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.0, 0.0, 0.0]
    AUC = 0.8

    plotting.plot_ROC (TPR, FPR, AUC)

    d = pd.DataFrame(np.random.randn(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
    plotting.plot_CM(d.corr())
    
    a=[0.98,0.2,0.1]
    plotting.plot_entropy(a)

