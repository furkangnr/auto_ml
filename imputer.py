# CODING : UTF-8

# DEVELOPER : furkan güner
# MAINTAINER : Furkan Güner & Ahmet Yüksel  ( 16.09.2022 - )


import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from scipy.stats import ttest_ind
from feature_engine.imputation import RandomSampleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class Imputer(BaseEstimator, TransformerMixin):
    
    # define __init__ method, which is a must for initializing any class instance.
    def __init__( self,
                  is_categoric_target = True,
                  features_list = None,
                  ordinal_features = None,
                  threshold = 0.95,
                  numerical_fill_value = -1,
                  high_na_drop = True,
                  add_na_indicator = True,
                  missing_class_label = "Missing",
                  apply_encoding = True,
                  random_state = None):
            
        
        assert type(is_categoric_target) == bool, "Only True or False accepted for is_categoric_target parameter."
        
        if features_list is not None:
            
            assert type(features_list) == list, "Please provide a list for feature_list parameter."
                
        assert type(ordinal_features) == list or ordinal_features is None," List or None for ordinal_features param."
        
        assert type(threshold) == float and 0 < threshold < 1, "Provide a float and it should between 0-1." 
                
        assert type(high_na_drop) == bool, "Only True or False accepted for high_na_drop parameter."
                
        assert type(add_na_indicator) == bool, "Only True or False accepted for add_na_indicator parameter."
        
        assert type(missing_class_label) == str, "Only string accepted for missing_class_label parameter."
        
        assert type(apply_encoding) == bool, "Only True or False accepted for apply_encoding parameter."
        
        if random_state is not None:
            assert type(random_state) == int, "Provide an integer value for random_state."
            
        
        # define class attributes.
        
        self.__version__ = "0.0.1"
        self.is_categoric_target = is_categoric_target
        self.threshold = threshold
        self.numerical_fill_value = numerical_fill_value
        self.high_na_drop = high_na_drop
        self.add_na_indicator = add_na_indicator
        self.missing_class_label = missing_class_label
        self.apply_encoding = apply_encoding
        self.features_list = features_list
            
        if random_state is None:
            self.random_state = 2022
        else:
            self.random_state = random_state
        
        if ordinal_features is None:
            self.ordinal_features = []
        else:
            self.ordinal_features = ordinal_features
        
        self.nominals = []
        self.numericals = []
        self.convert_numerical = []

    
    def type_assigner(self, df):
        
        # this method assigns variable types based on dtypes. It is used in .fit() method.
        
        for col in df.columns.tolist():
        
            if df[col].dtype == object:
            
                try:
                    df[col] = df[col].astype(float)
                    self.numericals.append(col)
                    self.convert_numerical.append(col)
                    
                except:
                    self.nominals.append(col)
            
            else:
                self.numericals.append(col)
                
        ordinals_in_nominals = [col for col in self.nominals if col in self.ordinal_features]
        
        for col in ordinals_in_nominals:
            
            self.nominals.remove(col)
        
        print("-----------------------------------------------------------------------------")
        print("Type assignment done ! Check .ordinal_features, .nominals and .numericals attributes.")
        print("-----------------------------------------------------------------------------")
        
        return None

    
    def null_ratio_calc(self,df,col):
        return df[col].isnull().mean()
       
    def binary_converter(self,df,col):
        
        na_array = np.where(df[col].isnull(), 1, 0)
            
        return na_array
            
    def nominal_converter(self,df,col,target_col,missing_class_label):
        
        df[col].fillna(missing_class_label, inplace = True)

        mapping_dict = df.groupby(col)[target_col].mean().to_dict()

        encoding_method = "Target Mean Encoding"
                    
        return mapping_dict, encoding_method

    def carried_information_for_target_categorical(self,df,col,target_col):
        
        boolean_mask = df[col].isnull()
        
        chi2_score, p_value = chi2(np.array(boolean_mask).reshape(-1, 1), df[target_col])
        
        return p_value
    
    def carried_information_for_target_numerical(self,df,col,target_col):
        
        boolean_mask = df[col].isnull()
    
        t_stats, p_value = ttest_ind(np.array(boolean_mask).reshape(-1, 1), df[target_col])
        
        return p_value
        
    
    def fit(self, X, y):
        
        start = time.time()
        
        assert type(X) == pd.DataFrame, "Provide a pandas dataframe object for 'X' parameter."
        assert type(y) == pd.Series, "Provide a pandas series object for 'y' parameter."
        assert y.name not in X.columns, "Input dataframe (X) should not contain target variable (y). Plase check."
        assert X.shape[0] == y.shape[0], "Input and target objects should have the same number of observations."
        
        if self.features_list is None:
            self.features_list = X.columns.tolist()
        
        for feature in self.features_list:
            assert feature in X.columns, f"Dataframe does not contain  : {feature}. Please check."
            
        self.input_dataframe = X
        self.high_na_cols = []
        self.na_indicator_cols = []
        self.impute_dict_for_cols = {}
        self.encode_dict_for_cols = {}
        self.random_imputer_cols = {}
        #self.other_cols = []
        
        target_col = y.name 
        
        df = X.copy()
        
        df = df[self.features_list]
        
        self.type_assigner(df)  
        
        try:
            df[target_col] = y.values
            
        except:
            raise Exception("Could not add target variable (y) to the input dataframe (X).")
            
        
        for col in self.features_list:
            
            print(" ------------ " , col, " ------------ ", "\n")

            null_ratio = self.null_ratio_calc(df,col)
            
            if null_ratio >= self.threshold: 
            
                print("--->>> ",col, f" has MR : {null_ratio}. Thus, will be converted to binary flag",
                         "If add_na_indicator = True.")
                
                if self.add_na_indicator:
                
                    self.na_indicator_cols.append(col)
                
                    print(f"--->>> {col} is added to na_indicator_cols list.")
                
                if self.high_na_drop:
                    
                    self.high_na_cols.append(col)
                    
                    print(f"--->>> {col} is added to high_na_cols list.")
                    
                else:
                    # yüksek NaN columnu droplamayacaksak eğer, içine missing_class_label dolduruyoruz.
                    # encoding listesine almıcaz ama.
                    self.impute_dict_for_cols.update( {col : self.missing_class_label} )
            
            else:
                
                if col in self.nominals:
                    
                    mapping_dict, encoding_method = self.nominal_converter(df,
                                                                           col,
                                                                           target_col,
                                                                           self.missing_class_label)
                    
                    print("--->>> ", col, f" is NOMINAL. We will assing a different class label for the missing values.")
                    
                    self.impute_dict_for_cols.update( {col : self.missing_class_label} )
                    
                    print("--->>> ", col, f": Missing values replaced with: {self.missing_class_label}.",
                          f"We will do:{encoding_method}.", "If apply_encoding =  True, encoding will be applied.")
                    
                    self.encode_dict_for_cols.update( { col : mapping_dict } )
                    print(f"{col} --- imputation and encoding values saved in seperate dictionaries.")
                    
                
                if col in self.ordinal_features or col in self.numericals:
                    
                    # DETERMINE BEING MISSING CARRIES INFORMATION FOR THE TARGET VARIABLE
                    
                    if self.is_categoric_target == True:
                        
                        
                        print("--->>> ", col, " Since target is categorical, we will apply chi-square test",
                              "in order to check whether missingness carries information for the target or not.", "\n")
                        
                        p_value = self.carried_information_for_target_categorical(df,col,target_col)
                        
            
                        print("--->>> ", col, f"chi-square test applied. P_value is : {p_value}", "\n")                    
                
                
                    elif self.is_categoric_target == False:
                    
                        print("--->>> ", col, " Since target is numerical, we will apply  T-test in order to check",
                        "whether missingness carries information for the target or not.", "\n")
                        
                        p_value = self.carried_information_for_target_numerical(df,col,target_col)
                        
                        print("--->>> ", col, f"T-test applied. P_value is : {p_value}.", "\n")
                
                
                    if p_value < 0.05 and col in self.ordinal_features:
                        
                        print(f"p_value < 0.05 and {col} is ORDINAL. We will create a binary flag for missingness",
                                "we will assing a different class label for the missing values.",
                                  "Finally, we will do target encoding.")
                
                        
                        self.na_indicator_cols.append(col)
                        
                        print(f"{col} is added to na_indicator_cols list.")
                        
                        self.impute_dict_for_cols.update( {col : self.missing_class_label} )
                        
                        mapping_dict, encoding_method = self.nominal_converter(df,col,target_col,
                                                                               self.missing_class_label)
                        
                        print("--->>> ", col, f": Missing values replaced with: {self.missing_class_label}.",
                              f"We will do {encoding_method}.", "If apply_encoding =  True, encoding will be applied.")
    
                        self.encode_dict_for_cols.update( { col : mapping_dict } )
                    
                        print("--->>> ", f"{col} --- imputation and encoding values saved in seperate dictionaries.")

                    
                    if p_value < 0.05 and col in self.numericals:
                        
                        print(f"p_value < 0.05 and {col} is NUMERICAL.")
                        
                        if null_ratio >= 0.75 or null_ratio <= 0.25:
                            
                            print(f"{col} has MR : {null_ratio}.",
                                  " We will add binary flag ( if add_na_indicator = True ), impute with median.")
                            
                            self.na_indicator_cols.append(col)
                            
                            print(f"{col} is added to na_indicator_cols list.")
                            
                            median_value = df[col].median()
                            
                            self.impute_dict_for_cols.update( {col : median_value } )
                            
                            print(f"{col} will be imputed with :  {median_value}.",
                                  "Stored this value in impute_dict_for_cols dictionary.")
                            
                        if 0.25 < null_ratio < 0.75:
                            
                            #  önceden self.other_cols attribute içinde tutuluyordu.
                            
                             print(f"{col} has MR : {null_ratio}." , 
                                   " We will fill NaN values with given value by numerical_fill_value.")
                            
                             self.impute_dict_for_cols.update( {col : self.numerical_fill_value } )
                             
                             print(f"{col} will be imputed with :  {self.numerical_fill_value}.",
                                  "Stored this value in impute_dict_for_cols dictionary.")
                                
                            #
                  
                           
                    elif p_value > 0.05:
                        
                        if null_ratio < 0.25 or null_ratio >= 0.75:
                            
                            if col in self.ordinal_features:
                                
                                print(f"p_value > 0.05. MR is : {null_ratio}  and {col} is ORDINAL.", "\n")
                                print(f"We will do mode imputation for {col}. Then, we will target encode.")
                                
                                mode_value = df[col].mode()
                                
                                df[col].fillna(mode_value, inplace=True)
                                
                                self.impute_dict_for_cols.update( {col : mode_value } )
                                
                                print("--->>> ",col, f": Missing values replaced with : {mode_value}. ",
                                      "We will do target encoding.")
                                    
                                mapping_dict = df.groupby(col)[target_col].mean().to_dict()
                                
                                self.encode_dict_for_cols.update( { col : mapping_dict } )
                                
                                print(f"{col} --- imputation and encoding values saved in seperate dictionaries.")
                                
                    
                            if col in self.numericals:
                                
                                print(f"p_value > 0.05. MR is : {null_ratio} and {col} is NUMERICAL.", "\n")
                                print(f"We will do median imputation for {col}.")
                                
                                median_value = df[col].median()
                                
                                print("--->>> ",col, f": Missing values will be replaced with {median_value}.")
                                
                                self.impute_dict_for_cols.update( {col : median_value } )
                                

                        elif 0.25 <= df[col].isnull().mean() < 0.75:
                                
                            print(f"{col} has MR : {null_ratio}. We will do RandomSampleImputation for {col}.", "\n")
                            
                            random_imputer = RandomSampleImputer(variables=[col], 
                                                                 random_state= self.random_state)
                            
                            random_imputer_fitted_object = random_imputer.fit(df[[col]])
                            
                            self.random_imputer_cols.update( { col : random_imputer_fitted_object } )
                            
                            print(f"{col} is added to random_imputer_cols dictionary.")
                            
                            if col in self.ordinal_features:
                                
                                print(f"{col} is ORDINAL, we will create an encoding dictionary.")
                                
                                mapping_dict = df.groupby(col)[target_col].mean().to_dict()
                                
                                self.encode_dict_for_cols.update( { col : mapping_dict } )
                                
                                print(f"{col} --- imputation and encoding values saved in seperate dictionaries.",
                                     "Since we used random sample imputer object, we saved fitted object in another",
                                      "dictionary in random_imputer_cols attribute.")
                                
        end = time.time()
            
        elapsed = end - start 
            
        print("\n", f"Fitting process completed in {elapsed} seconds. ", "\n")
        
        return self
    
    
    def transform(self, X):
        
        start = time.time()
        
        assert type(X) == pd.DataFrame, "Provide a pandas dataframe object for 'X' parameter."
        
        for feature in self.features_list:
            assert feature in X.columns, f"Dataframe does not contain  : {feature}. Please check."
        
        for col in X.columns:
            if col not in self.features_list:
                raise Exception(f"{col} has not seen during fit. Check input dataframe (X) columns...")
                
        assert X.shape[1] == self.input_dataframe.shape[1], "shape mismatch between input dataframes in fit & transform."
        
        if len(self.convert_numerical) != 0:
            
            for col in self.convert_numerical:
                
                X[col] = X[col].astype(float)
                
        
        if self.apply_encoding == True:
            
            # new label check ! 
            
            categoricals = self.nominals + self.ordinal_features
            
            for col in categoricals:
                
                labels = X[col].dropna().unique().tolist()
                
                for label in labels:
                    
                    if label not in self.input_dataframe[col].dropna().unique().tolist():
                        
                        print(f"New label observed in {col}.\
                                In order to do encoding, we need to have encoding value for new label in {col}.\
                                New Label is : {label}.", "\n")
                        
                        print(f"For now, we will update encoding dictionary for {col}  with {label : np.nan}")
                        
                        self.encode_dict_for_cols[col].update( {label : np.nan} )
                   
    
        df = X.copy()
        
        
        if self.add_na_indicator:
            
            print(f"NA indicator addition will be applied to ------->>> {self.na_indicator_cols}.", "\n")
            
            for col in self.na_indicator_cols:
                
                df[col + "_NA"] = np.where(df[col].isnull(), 1, 0)
                
                print(f"{col}_NA added to the dataframe.", "\n")
        
        if self.high_na_drop:
            
            df.drop(self.high_na_cols, axis = 1, inplace = True)
            
            print(f"High MR columns dropped. These are -------->>>   {self.high_na_cols}.", "\n")
            
        
        print("\n", "IMPUTATION started on these columns ------>>>", list(self.impute_dict_for_cols.keys()), "\n" )
        
        for col in self.impute_dict_for_cols.keys():
            
            df[col].fillna(self.impute_dict_for_cols[col], inplace = True)
            
            print(col, " :  imputed with  ------>>>" , self.impute_dict_for_cols[col], "\n")
            
        
        for col in self.random_imputer_cols.keys():
            
            print("Random Imputation applied on these columns ------>>>", list(self.random_imputer_cols.keys()), "\n")
        
            try: 
                
                df[col] = self.random_imputer_cols[col].transform(df[col])  #to-do : içine data verilmemiş ? 
                
            except:
                
                raise Exception("Random Imputation step has failed.")
                
            
        if self.apply_encoding:
            
            print("\n", "ENCODING started on these columns ------>>>", list(self.encode_dict_for_cols.keys()), "\n" )
            
            for col in self.encode_dict_for_cols.keys():
                
                df[col] = df[col].map(self.encode_dict_for_cols[col])
                
                print(col, "encoded with :", self.encode_dict_for_cols[col], "\n" )
            
        
        #print("\n", "NO TRANSFORMATION APPLIED ON THESE COLUMNS ------>>>", self.other_cols, "\n")
        
        end = time.time()
        
        elapsed = end - start
        
        print(f"Transformation has completed in {elapsed} seconds.")
        
        return df
    
    def fit_transform(self, X, y):
        
        self.fit(X,y)
        
        df_transformed = self.transform(X)
        
        return df_transformed

