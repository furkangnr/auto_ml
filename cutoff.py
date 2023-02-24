# TO-DO :  1) Custom metric definitions. 

# 2) Bir benzer akışı -- regression taskleri için düşünebilir miyiz ? 


# KAPSAMLI BIR DOCSTRING LAZIM ! 

# todo : DRY ( Do not Repeat Yourself )

# assertionlar eklenicek.

# merve : görseller ( interactive dashboard )
# taha : recommend ( kesme arrayi akılcı bir şekilde recommend edilecek )
# ahmet : segment bazlı & multi-class.



import pandas as pd
import numpy as np


from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import auc, precision_recall_curve, f1_score, precision_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score, cohen_kappa_score


def pr_auc_opt(y_train, probas):
    
    precision_, recall_, proba_ = precision_recall_curve(y_train, probas)
    
    optimal_proba_cutoff_pr_auc = sorted(list(zip(np.abs(precision_ - recall_), proba_)), 
                                         key=lambda i: i[0], reverse=False)[0][1]
    
    return round(optimal_proba_cutoff_pr_auc, 6)


def roc_auc_opt(y_train, probas):
    
    false_pos_rate, true_pos_rate, proba = roc_curve(y_train, probas)
    
    optimal_proba_cutoff_roc = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), 
                                      key=lambda i: i[0], reverse=True)[0][1]
    
    return round(optimal_proba_cutoff_roc, 6)


def pr_auc(y_true, y_pred):
    
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    
    pr_auc = auc(recall, precision)
    
    return pr_auc 


def plot_precision_recall(results_all_opt):
    
    style.use("fivethirtyeight")

    fig, ax = plt.subplots(1,1, figsize = (12,8))
    
    ax.plot(results_all_opt.index, results_all_opt["precision"], color = "g", label = "precision")
    
    ax.plot(results_all_opt.index, results_all_opt["recall"], color = "b", label = "recall")
    
    ax.plot(results_all_opt.index, results_all_opt["f1_score"], color = "r", label = "f1_score")
    
    ax.set_xticks(results_all_opt.index)
    
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    ax.vlines(x = results_all_opt["f1_score"].idxmax(), ymin = 0, ymax = 1, linestyles = "dashed", label = "opt-f1")
    
    ax.text(x = results_all_opt["f1_score"].idxmax(), y = 0.9, 
        s = "opt f1_score proba = "+str(round(results_all_opt.loc[results_all_opt["f1_score"].idxmax(), "Cut-Off"],6)), 
        fontsize = "large")
    
    ax.set_title("Cut-OFF Precision - Recall Graph", fontsize = "x-large", color ="k")
    
    ax.set_xlabel("Cut-OFF Points", fontsize = "x-large", color ="k")
    
    ax.set_ylabel("Metric score value (0-1)",  fontsize = "x-large", color ="k")
    
    ax.legend()
    
    plt.tight_layout()
    
    plt.savefig("Precision_Recall.png")
    
    plt.show()



def cutoff_finder(data, target, probas, ratio = [1,2,3,4,5], reference_target_ratio = 0.0028, opt_cut_off = None):
        
        data_sorted = data.sort_values(probas, ascending = False) # probaları yukardan aşağıya olacak şekilde sırala
        
        all_fraud_count = Counter(data_sorted[target])[1]   # datadaki toplam 1 sayısı. 
        
        all_fraud_ratio = data_sorted[target].mean()  # data target ratio 
        
        total_obs = data_sorted.shape[0]   # total number of observations
        
        recall_scores = []
        precision_scores = []
        lift_scores = []
        adjusted_lift_scores = []
        f1_scores = []
        fbeta_05_scores = []
        fbeta_2_scores = []
        cohen_kappa_scores = []
        #roc_auc_scores = []
        single_roc_auc_scores = []
        #pr_auc_scores = []
        cut_off_probas = []
        observations_count = []
        single_pr_auc_scores = []
        index = []
        
        
        optimal_proba_cutoff_pr_auc = pr_auc_opt(data[target], data[probas])
        
        optimal_proba_cutoff_roc = roc_auc_opt(data[target], data[probas])
        
        print('Optimal Cutoff Probability based on PR Curve:', optimal_proba_cutoff_pr_auc)
        print('Optimal Cutoff Probability based on ROC Curve:', optimal_proba_cutoff_roc)
        
      
        if opt_cut_off is not None:
            
            num = data[data[probas] >= opt_cut_off].shape[0]
            
            subset_fraud_count = data_sorted[:num][target].sum()
            subset_fraud_ratio = data_sorted[:num][target].mean()
                            
            recall = subset_fraud_count / all_fraud_count
            lift = subset_fraud_ratio / all_fraud_ratio
            adjusted_lift = subset_fraud_ratio / reference_target_ratio
            proba = data_sorted[:num][probas].tail(1).values.tolist()[0]
            
            s1 = pd.Series(np.ones(num))
            s2 = pd.Series(np.zeros(total_obs - num))
            s3 = s1.append(s2)
            
            f1 = f1_score(data_sorted[target], s3)
            fbeta_05 = fbeta_score(data_sorted[target], s3, beta = 0.5)
            fbeta_2 = fbeta_score(data_sorted[target], s3, beta = 2)
            cohen_kappa = cohen_kappa_score(data_sorted[target], s3)
            #pr_auc_score = pr_auc(data_sorted[target], s3)
            #roc_auc = roc_auc_score(data_sorted[target], s3)
            single_roc_auc = roc_auc_score(data_sorted[target], data_sorted[probas])
            single_pr_auc_score = pr_auc(data_sorted[target], data_sorted[probas])
            precision = precision_score(data_sorted[target], s3)
            
            recall_scores.append(recall)
            precision_scores.append(precision)
            lift_scores.append(lift)
            adjusted_lift_scores.append(adjusted_lift)
            f1_scores.append(f1)
            fbeta_05_scores.append(fbeta_05)
            fbeta_2_scores.append(fbeta_2)
            cohen_kappa_scores.append(cohen_kappa)
            #pr_auc_scores.append(pr_auc_score)
            #roc_auc_scores.append(roc_auc)
            single_pr_auc_scores.append(single_pr_auc_score)
            single_roc_auc_scores.append(single_roc_auc)
            
            cut_off_probas.append(proba)
            observations_count.append(num)
            
            matrix = confusion_matrix(data_sorted[target], s3)
            disp = ConfusionMatrixDisplay(matrix)
            disp.plot()
            
            plt.title(f"Opt-Cut-OFF, Observations:{num}.")
            plt.tight_layout()
            plt.show()
            
            index.append("OPT")
        
        for r in ratio:
            
            num = int(data_sorted.shape[0] * r  /  100)
                            
            subset_fraud_count = data_sorted[:num][target].sum()
            subset_fraud_ratio = data_sorted[:num][target].mean()
                            
            recall = subset_fraud_count / all_fraud_count
            lift = subset_fraud_ratio / all_fraud_ratio
            adjusted_lift = subset_fraud_ratio / reference_target_ratio
            proba = data_sorted[:num][probas].tail(1).values.tolist()[0]
                            
            s1 = pd.Series(np.ones(num))
            s2 = pd.Series(np.zeros(total_obs - num))
            s3 = s1.append(s2)
                
            f1 = f1_score(data_sorted[target], s3)
            fbeta_05 = fbeta_score(data_sorted[target], s3, beta = 0.5)
            fbeta_2 = fbeta_score(data_sorted[target], s3, beta = 2)
            cohen_kappa = cohen_kappa_score(data_sorted[target], s3)
            #pr_auc_score = pr_auc(data_sorted[target], s3)
            #roc_auc = roc_auc_score(data_sorted[target], s3)
            single_roc_auc = roc_auc_score(data_sorted[target], data_sorted[probas])
            single_pr_auc_score = pr_auc(data_sorted[target], data_sorted[probas])
            precision = precision_score(data_sorted[target], s3)
                
            recall_scores.append(recall)
            lift_scores.append(lift)
            adjusted_lift_scores.append(adjusted_lift)
            f1_scores.append(f1)
            fbeta_05_scores.append(fbeta_05)
            fbeta_2_scores.append(fbeta_2)
            cohen_kappa_scores.append(cohen_kappa)
            #pr_auc_scores.append(pr_auc_score)
            #roc_auc_scores.append(roc_auc)
            single_pr_auc_scores.append(single_pr_auc_score)
            single_roc_auc_scores.append(single_roc_auc)
            precision_scores.append(precision)
            cut_off_probas.append(proba)
            observations_count.append(num)
            index.append(str(np.around(r,2)))
                
            # ConfusionMatrixDisplay.from_predictions(data_sorted[target], s3, ax = axes[i][counter] )

            matrix = confusion_matrix(data_sorted[target], s3)
            disp = ConfusionMatrixDisplay(matrix)
            disp.plot()
                
            plt.title(f"Ratio : {np.around(r,2)}, Observations : {num}.")
            plt.tight_layout()
            plt.show()
                
            #print("\n")
            
                
        results = { "ratio" : index,
                    "Cut-Off" : cut_off_probas,
                    "Observations" : observations_count,
                    "recall" : recall_scores,
                    "precision" : precision_scores,
                    "f1_score" : f1_scores,
                    "f_05_score" : fbeta_05_scores,
                    "f_2_score" : fbeta_2_scores,
                    "cohen_kappa" : cohen_kappa_scores,
                    #"roc_auc_scores" : roc_auc_scores,
                    "single_roc_auc" : single_roc_auc_scores,
                    "lift" : lift_scores,
                    "adjusted_lift" : adjusted_lift_scores,
                    #"pr_auc_scores" : pr_auc_scores,
                    "Single_pr_auc" : single_pr_auc_scores,
                    #"pr_auc" : pr_auc_scores
                  }
            
        df = pd.DataFrame(results).set_index("ratio")
        
        df["True_Positive_Rate"] = df["precision"]
        
        df["False_Positive_Rate"] = 1 - df["precision"]
        
        plot_precision_recall(df)
        
        style.use("classic")

        num2 = df.loc[df.index == df["f1_score"].idxmax(), "Observations"]
        
        s4 = pd.Series(np.ones(num2))
        s5 = pd.Series(np.zeros(total_obs - num2))
        s6 = s4.append(s5)
        
        matrix = confusion_matrix(data_sorted[target], s6)
        disp = ConfusionMatrixDisplay(matrix)
        disp.plot()
        
        f1_opt_proba = round(df.loc[df.index == df["f1_score"].idxmax(),"Cut-Off"].values[0], 6)
                
        plt.title(f'f1_score OPT, Probability: {f1_opt_proba}.')
        plt.tight_layout()
        plt.show()
        
        opt_cut_off_probas = { "pr_auc_opt" :  optimal_proba_cutoff_pr_auc, "roc_auc_opt" : optimal_proba_cutoff_roc }
        
        for opt_cut_off in opt_cut_off_probas.keys():
            
            proba = opt_cut_off_probas[opt_cut_off]
            
            num = data[data[probas] >= proba].shape[0]
            
            subset_fraud_count = data_sorted[:num][target].sum()
            subset_fraud_ratio = data_sorted[:num][target].mean()
                            
            recall = subset_fraud_count / all_fraud_count
            lift = subset_fraud_ratio / all_fraud_ratio
            adjusted_lift = subset_fraud_ratio / reference_target_ratio
            proba = data_sorted[:num][probas].tail(1).values.tolist()[0]
            
            s1 = pd.Series(np.ones(num))
            s2 = pd.Series(np.zeros(total_obs - num))
            s3 = s1.append(s2)
            
            f1 = f1_score(data_sorted[target], s3)
            fbeta_05 = fbeta_score(data_sorted[target], s3, beta = 0.5)
            fbeta_2 = fbeta_score(data_sorted[target], s3, beta = 2)
            cohen_kappa = cohen_kappa_score(data_sorted[target], s3)
            #pr_auc_score = pr_auc(data_sorted[target], s3)
            #roc_auc = roc_auc_score(data_sorted[target], s3)
            single_roc_auc = roc_auc_score(data_sorted[target], data_sorted[probas])
            single_pr_auc_score = pr_auc(data_sorted[target], data_sorted[probas])
            precision = precision_score(data_sorted[target], s3)
            
            recall_scores.append(recall)
            precision_scores.append(precision)
            lift_scores.append(lift)
            adjusted_lift_scores.append(adjusted_lift)
            f1_scores.append(f1)
            fbeta_05_scores.append(fbeta_05)
            fbeta_2_scores.append(fbeta_2)
            cohen_kappa_scores.append(cohen_kappa)
            #pr_auc_scores.append(pr_auc_score)
            #roc_auc_scores.append(roc_auc)
            single_pr_auc_scores.append(single_pr_auc_score)
            single_roc_auc_scores.append(single_roc_auc)
            cut_off_probas.append(proba)
            observations_count.append(num)
            
            style.use("classic")
            
            matrix = confusion_matrix(data_sorted[target], s3)
            disp = ConfusionMatrixDisplay(matrix)
            disp.plot()
            
            plt.title(f"{opt_cut_off}, Proba:{round(proba,6)}.")
            plt.tight_layout()
            plt.show()
            
            num_to_index = round((num / data.shape[0]) * 100,2)
            
            index.append(f"{opt_cut_off} : {num_to_index}")
            
            
        results = { "ratio" : index,
                    "ratio2": index,
                    "Cut-Off" : cut_off_probas,
                    "Observations" : observations_count,
                    "recall" : recall_scores,
                    "precision" : precision_scores,
                    "f1_score" : f1_scores,
                    "f_05_score" : fbeta_05_scores,
                    "f_2_score" : fbeta_2_scores,
                    "cohen_kappa" : cohen_kappa_scores,
                    #"roc_auc_scores" : roc_auc_scores,
                    "single_roc_auc" : single_roc_auc_scores,
                    "lift" : lift_scores,
                    "adjusted_lift" : adjusted_lift_scores,
                    #"pr_auc_scores" : pr_auc_scores,
                    "Single_pr_auc" : single_pr_auc_scores,
                    #"pr_auc" : pr_auc_scores
                  }
            
        df = pd.DataFrame(results).set_index("ratio")
        
        df["True_Positive_Rate"] = df["precision"]
        
        df["False_Positive_Rate"] = 1 - df["precision"]
        
        df.sort_values("Cut-Off", ascending = False, inplace = True)
        
        df.loc[df["ratio2"] == df["f1_score"].idxmax(), "ratio2"] = "f1_max"

        df.loc[df["ratio2"] == df["f_05_score"].idxmax(), "ratio2"] = "f_05_max"

        df.loc[df["ratio2"] == df["f_2_score"].idxmax(), "ratio2"] = "f2_max"
        
        df.reset_index(inplace = True)
                  
        df.set_index("ratio2", inplace = True)
        
        df.to_excel("Precision_Recall_Optimization.xlsx")
        
        return df, f1_opt_proba
    

def test_performance(data, target, probas, opt_cut_off_from_dev = 0.9134, reference_target_ratio = 0.0028):
    
        # bu fonksiyon her ihtimale karşı buraya koyuldu, dev datası üzerinden yukardaki fonksiyon çalıştırılarak 
        # opt-cut-off belirlenir. Sonra o değer burdaki parametreye verilir. 
        # Trivial olmasının sebebi, yukardaki fonksiyonu 2 defa çalıştırarak
        # aynı sonucu elde edebiliyor olmamız. İlk çalıştırıldığında (dev datası üzerinden) opt_cut_off parametresi None 
        # verilir.
        # onun çıktıları analiz edilerek seçilen cut-off değeri ikinci çalıştırıldığında ( test datası üzerinden ) 
        # opt_cut_off parametresine verilir. Benzer sonuç alınacaktır. 
    
        data_sorted = data.sort_values(probas, ascending = False) # probaları yukardan aşağıya olacak şekilde sırala
        
        all_fraud_count = Counter(data_sorted[target])[1]   # datadaki toplam 1 sayısı. 
        
        all_fraud_ratio = data_sorted[target].mean()  # target ratio 
        
        total_obs = data_sorted.shape[0]   # total number of observations
        
        recall_scores = []
        precision_scores = []
        lift_scores = []
        adjusted_lift_scores = []
        f1_default = []
        f1_scores = []
        fbeta_05_scores = []
        fbeta_2_scores = []
        cohen_kappa_scores = []
        #roc_auc_scores = []
        single_roc_auc_scores = []
        #pr_auc_scores = []
        cut_off_probas = []
        observations_count = []
        single_pr_auc_scores = []
        index = []
        
        num = data[data[probas] >= opt_cut_off_from_dev].shape[0]
        num2 = data[data[probas] >= 0.5].shape[0]
            
        subset_fraud_count = data_sorted[:num][target].sum()
        subset_fraud_ratio = data_sorted[:num][target].mean()
                            
        recall = subset_fraud_count / all_fraud_count
        lift = subset_fraud_ratio / all_fraud_ratio
        adjusted_lift = subset_fraud_ratio / reference_target_ratio
        proba = data_sorted[:num][probas].tail(1).values.tolist()[0]
            
        s1 = pd.Series(np.ones(num))
        s2 = pd.Series(np.zeros(total_obs - num))
        s3 = s1.append(s2)
        
        s4 = pd.Series(np.ones(num2))
        s5 = pd.Series(np.zeros(total_obs - num2))
        s6 = s4.append(s5)
        
            
        f1 = f1_score(data_sorted[target], s3)
        f1_default_score = f1_score(data_sorted[target], s6)
        fbeta_05 = fbeta_score(data_sorted[target], s3, beta = 0.5)
        fbeta_2 = fbeta_score(data_sorted[target], s3, beta = 2)
        cohen_kappa = cohen_kappa_score(data_sorted[target], s3)
        #pr_auc_score = pr_auc(data_sorted[target], s3)
        #roc_auc = roc_auc_score(data_sorted[target], s3)
        single_roc_auc = roc_auc_score(data_sorted[target], data_sorted[probas])
        single_pr_auc_score = pr_auc(data_sorted[target], data_sorted[probas])
        precision = precision_score(data_sorted[target], s3)
            
        recall_scores.append(recall)
        precision_scores.append(precision)
        lift_scores.append(lift)
        adjusted_lift_scores.append(adjusted_lift)
        f1_scores.append(f1)
        f1_default.append(f1_default_score)
        fbeta_05_scores.append(fbeta_05)
        fbeta_2_scores.append(fbeta_2)
        cohen_kappa_scores.append(cohen_kappa)
        #pr_auc_scores.append(pr_auc_score)
        #roc_auc_scores.append(roc_auc)
        single_roc_auc_scores.append(single_roc_auc)
        single_pr_auc_scores.append(single_pr_auc_score)
        cut_off_probas.append(proba)
        observations_count.append(num)
        
        style.use("classic")
            
        matrix = confusion_matrix(data_sorted[target], s3)
        disp = ConfusionMatrixDisplay(matrix)
        disp.plot()
            
        plt.title(f"Opt-Cut-OFF, Observations:{num}.")
        plt.tight_layout()
        plt.show()
            
        index.append("OPT")
        
        results = { "ratio" : index,
                    "Cut-Off" : cut_off_probas,
                    "Observations" : observations_count,
                    "recall" : recall_scores,
                    "precision" : precision_scores,
                    "f1_default" : f1_default,
                    "f1_score_optimized" : f1_scores,
                    "f_05_score" : fbeta_05_scores,
                    "f_2_score" : fbeta_2_scores,
                    "cohen_kappa" : cohen_kappa_scores,
                    "lift" : lift_scores,
                    "adjusted_lift" : adjusted_lift_scores,
                    #"pr_auc_optimized" : pr_auc_scores,
                    "Single_pr_auc" : single_pr_auc_scores,
                    #"roc_auc_optimized" : roc_auc_scores,
                    "Single_roc_auc" : single_roc_auc_scores
                  }
            
        df = pd.DataFrame(results).set_index("ratio")
        
        df["True_Positive_Rate"] = df["precision"]
        
        df["False_Positive_Rate"] = 1 - df["precision"]
        
        return df 
