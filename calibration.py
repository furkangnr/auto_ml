import time 
import os 
import joblib
from datetime import datetime
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns 

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


# PATH SETTING FOR OUTPUTS 
now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

if os.path.exists("./plots") == False:
        
        os.makedirs("./plots")
        
        os.makedirs(f"./plots/{now}")
        
else:
        os.makedirs(f"./plots/{now}")
        
if os.path.exists("./models") == False:
        
    os.makedirs("./models")
        
    os.makedirs(f"./models/{now}")
        
else:
    os.makedirs(f"./models/{now}")


def expected_calibration_error(y_true, probas, bins = "fd"):
    
    bin_count, bin_edges = np.histogram(probas, bins = bins)
    n_bins = len(bin_count)
    bin_edges[0] -= 1e-8
    bin_id = np.digitize(probas, bin_edges, right = True) - 1
    bin_ysum = np.bincount(bin_id, weights = y_true, minlength = n_bins)
    bin_probasum = np.bincount(bin_id, weights = probas, minlength = n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    ece = np.abs(( bin_probamean - bin_ymean ) * bin_count).sum() / len(probas)
    
    return round(ece,4)


def plot_calibration(y_true, probas, label, ece_score, n_bins = 10, strategy = "uniform"):

    proba_true, proba_pred = calibration_curve(y_true, probas, n_bins = n_bins, strategy = strategy)
    
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')

    plt.plot(proba_pred, proba_true, label = label)
    
    plt.title(f"Expected Calibration Error is : {ece_score}.")

    plt.legend()
    
    plt.savefig(f"./plots/{now}/{label}.png")
    
    plt.show()  # should be removed from production code.


def calibrator(data):
    
    assert type(data) == dict, "Provide a dictionary object."
    assert len(data) > 2, "Please provide at least 3 key-value pairs."
    assert "train" in data.keys(), "Dictionary must have a key 'train'."
    assert "valid" in data.keys(), "Dictionary must have a key 'valid'."
    assert "test" in data.keys(), "Dictionary must have a key 'test'."
    
    for key in data.keys():
        
        assert type(data[key]) == list, f"Provide a list for dictionary values. Error comes from : {key}"
        assert len(data[key]) == 2, f"Value list should be length of 2. Error comes from : {key}"
        assert data[key][0].shape[0] == data[key][1].shape[0], f"Values should have equal lengths.Error comes from : {key}"
        assert type(data[key][0]) == type(data[key][1]) == pd.Series, f"Values should have type of pandas Series.\
        Error comes from : {key}."
        assert data[key][0].nunique() == 2, f"First item of dictionary values should be target variable. Must consist of\
        0-1 s. Error comes from : {key}."
        assert 0 <= data[key][1].min() and  data[key][1].max() <= 1, f"Second item of dictionary values should be probas.\
        Error comes from : {key}."

        
    model_valid = IsotonicRegression(y_min = 0 , y_max = 1, out_of_bounds = "clip")
    
    model_test = IsotonicRegression(y_min = 0 , y_max = 1, out_of_bounds = "clip")
    
    ece_scores = {} 
    
    for key in data.keys():
        
        ece =  expected_calibration_error(data[key][0], data[key][1])
        
        ece_scores.update( { f"{key}_uncalibrated" : ece } )
        
    model_valid.fit(data["valid"][1], data["valid"][0])
    
    model_test.fit(data["test"][1], data["test"][0])
    
    calibrated_probas = {} 
    
    for key in data.keys():
        
        if key != "valid":
            
            calibrated = model_valid.predict(data[key][1])
            
            calibrated_probas.update( { key : calibrated } )
            
            ece =  expected_calibration_error(data[key][0], calibrated)
            
            ece_scores.update( { f"{key}_calibrated" : ece } )
            
            
        else: 
            
            calibrated = model_test.predict(data[key][1])
            
            calibrated_probas.update( { key : calibrated } )
            
            ece =  expected_calibration_error(data[key][0], calibrated)
            
            ece_scores.update( { f"{key}_calibrated" : ece } )
            
    
    for key in data.keys():
        
        print(f"------------------------ {key.upper()} UNCALIBRATED PLOT ---------------------------------")
        
        plot_calibration(data[key][0], data[key][1] , label = f"uncalibrated_{key}", 
                         ece_score = ece_scores[f"{key}_uncalibrated"],
                         n_bins = 10, strategy = "uniform")
        
        print(f"------------------------ {key.upper()} CALIBRATED PLOT -----------------------------------")
        
        plot_calibration(data[key][0], calibrated_probas[key], label = f"calibrated_{key}", 
                         ece_score = ece_scores[f"{key}_calibrated"],
                         n_bins = 10, strategy = "uniform")
        
        print("-------------------------------------------------------------------------------------------", "\n", "\n")
        
        
    joblib.dump(model_valid, f"./models/{now}/calibrator_valid.joblib")
    
    joblib.dump(model_test, f"./models/{now}/calibrator_test.joblib")
    
    return model_valid, model_test, calibrated_probas, ece_scores 