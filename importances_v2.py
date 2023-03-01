# çıktıları alma seçeneğini ekle. export = True, feature_importances diye bir klasöre timestampli halde atsın...


import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt


class Importances:

    
    def __init__(self,
                 model,
                 X_train,
                 y_train,
                 n_features = 50):
    
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_features = n_features
        
        
    def fit(self):
     
        # fit data to the given model.
        
        try:
            self.model.fit(self.X_train, self.y_train)
        
        except:
            raise Exception("Could not fit given data to the given model. Pease check input data or model again.")
    
    
    def shap_values(self):
        
        # fit the fitted model to the shap tree explainer object.
    
        explainer_obj = shap.TreeExplainer(self.model)
        
        # obtain shap values from explainer, using data which is fitted to the model before ( train set )
    
        shap_values = explainer_obj.shap_values(self.X_train)
        
        self.shap_values = shap_values
        
        shap.summary_plot(self.shap_values, features = self.X_train, max_display = self.n_features,
                          show = False, plot_size = (10,16))
        
        plt.tight_layout()
        
        plt.savefig(f"shap_original_plot_top_{self.n_features}_features.png")
        
        plt.show()
        
        shap.summary_plot(self.shap_values, features = self.X_train, plot_type = "bar", max_display = self.n_features, 
                          show = False, plot_size = (10,16))
        
        plt.tight_layout()
        
        plt.savefig(f"shap_bar_plot_top_{self.n_features}_features.png")
        
        plt.show()
    
        return None
    
    
    def feature_importances(self):
        
        # obtain feature importances scores using .feature_importances_ attribute of the model.
        
        df = pd.DataFrame(self.model.feature_importances_, index = self.X_train.columns)
        
        df.rename( columns = { 0 : "IMPORTANCE"} , inplace = True)
    
        df.sort_values("IMPORTANCE", ascending = False, inplace = True)
        
        df["CUMSUM"] = df["IMPORTANCE"].cumsum()
        
        df["GAIN_RATIO"] = df["CUMSUM"] / df["IMPORTANCE"].sum()
        
        self.importance_df = df
        
        self.importance_df["GAIN_RATIO"].reset_index().plot()
        
        plt.tight_layout()
        
        plt.savefig(f"model_importances_all_features.png")
        
        plt.show()
        
        return self.importance_df
    
    
    def final_feature_selection(self):
        
        shap_top_features = self.shap_impact_df.head(self.n_features).index.tolist()
        
        model_top_features = self.importance_df.head(self.n_features).index.tolist()
        
        self.shap_top_features = shap_top_features
        
        self.model_top_features = model_top_features
        
        common_features = [col for col in self.shap_top_features if col in self.model_top_features]
        
        self.common_features = common_features
        
        final_features = list(set(self.shap_top_features + self.model_top_features))
        
        self.final_features = final_features
        
        print("Selected top performant shap features have length : ", len(self.shap_top_features))
        
        print("Selected top performant model features have length : ", len(self.model_top_features))
        
        print("Common features have length : ", len(self.common_features))
        
        print("Final features have length : ", len(self.final_features))
        
        return self.final_features
        