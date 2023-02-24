import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.stats import sem, t
from numpy import mean

def bootstrapscores(model, data, target = "my_target", n_samples = 1000):
    """ 
        Definition :
        ------------
        This function takes a dataframe and creates its bootstrapped samples via sampling with replacement method. After creating 
        bootstrapped samples, function fits the model with respect to the target. Lastly, it evaluates models auc score. Desired number 
        of bootstrapped samples can be controlled via n_samples parameter.
        
        Parameters : 
        ------------
            model( object ) : Scikit-learn estimator object. Model should have .predict_proba() method.
            
            data( pd.DataFrame ) : Pandas dataframe including target variable and features.
            
            target ( string ) : Target column to be predicted.
            
            n_samples ( int ) : Desired number of bootstrapped samples.
            
        Returns : 
        ---------
            scores_list ( list ) : A list of auc scores. Each score belongs to different bootstrapped sample.
           
    """
    scores_list = []
    for i in range(n_samples):
        seed = i * 2 + 44
        sample_data = data.sample( n = len(data), replace = True, random_state = seed ) 
        
        features = sample_data.drop(target, axis = 1)
        
        y_true = sample_data.loc[:,target]
        y_pred = model.predict_proba(features)[:,1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        scores_list.append(roc_auc)
        
    return scores_list





def confidence(scores, interval = 0.95):
    """ 
        Definition : 
        ------------
        This function takes a list of auc scores and calculates gini scores based on that. After obtaining gini scores, it calculates           confidence intervals for different ratios. Finally it prints out the starting and ending points for respective confidence 
        interval. Ratio of confidence can be controlled by interval parameter.
        
        Parameters : 
        ------------
            scores ( list ) : A list of auc scores.
            
            interval ( float ) : Confidence interval ratio.
            
    """       
    df = pd.DataFrame(scores, columns = ["roc"])
    df["gini"] = df["roc"] * 2 - 1
    
    n = len(df["gini"])
    m = mean(df["gini"])
    std_err = sem(df["gini"])
    
    h = std_err * t.ppf( (1 + interval) / 2 , n-1)
    start = m - h
    end = m + h 
    
    print(f"Confidence Interval with alfa %{interval}", start,end );
    print("Median Gini: ",df[['gini']].median()[0])
    print("Mean Gini: ",df[['gini']].mean()[0])
    print("STD Gini: ",df[['gini']].std()[0])


        