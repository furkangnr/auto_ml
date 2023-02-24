# fit edilmiş grid search objesi verildiğinde mean cv skorlarına ve std cv skorlarına
# göre bir akıl kullanarak top performing modelleri
# listeye alır. Nihai modeli belirlemek adına bu listenin içerisindeki her modelin
# test datası üzerindeki performansına bakmak gerekir.

import pandas as pd
import copy
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error


def grid_optimize(empty_model, fitted_gs_object, top_n_mean=20, top_n_std=10):

    models = []
    
    grid_results = pd.DataFrame.from_dict(fitted_gs_object.cv_results_)
    
    grid_results.sort_values("rank_test_score", ascending=True, inplace=True)
    
    df = grid_results.head(top_n_mean).sort_values("std_test_score", ascending=True).head(top_n_std)
    
    for parameters in df["params"].values:
        
        params = eval(str(parameters))
        
        new_model = copy.deepcopy(empty_model.set_params(**params))
        
        models.append(new_model)
        
    return models


def test_performances(models, x_train, y_train, x_test, y_test, problem_type="classification", metric="roc_auc"):

    score = None

    results = {}
    
    counter = 1
    
    for model in models:
        
        # print(f"Model fitting has started. This is model number {counter}.", "\n")
        
        model.fit(x_train, y_train)
        
        if problem_type == "classification":
        
            preds = model.predict_proba(x_test)[:, 1]
        
            if metric == "roc_auc":
            
                score = roc_auc_score(y_test, preds)
            
            # if metric == .... # goes like this.
        
        if problem_type == "regression":
            
            preds = model.predict(x_test)
            
            if metric == "mean_absolute_percentage_error":
                
                score = mean_absolute_percentage_error(y_test, preds)
            
            # if metric == .... # goes like this.

        results.update({model: score})
        
        print(f"Model fitting finished for model number {counter}.")
        
        counter += 1

    df = pd.DataFrame.from_dict(results, orient="index").rename(columns={0: f"{metric}"})
    
    df.sort_values(f"{metric}", ascending=False, inplace=True)
    
    best_model = df.head(1).index.tolist()[0]
    
    return df, best_model
