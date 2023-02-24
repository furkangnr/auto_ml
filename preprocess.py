# TODO : Docstring, assertions, new processing methods can be added. ( duplicate checks for rows and columns etc. )

# encoding_type parametresi eklenebilir. ( şu an için gerek yok, encoding = False denilip encoder.py dosyasına yönlendirilebilir )

# row-wise da bakmak lazım ( row-wise NaN ratio yukarda olan columnlar drop ? )

# data-type için data dictionaryden de yararlanmak düşünülmeli. Numeric olarak okumuş pandas ancak aslında categoric ?? 
# plaka kodu, şube kodu gibi.



import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np  
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


class Processor(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 na_threshold = 0.9,
                 cardinality_threshold = 50,
                 drop_high_na_cols = True,
                 drop_zero_variance_features = True,
                 drop_high_cardinal_features = True,
                 encoding = True,
                 memory_reduction = True,
                 export = True
                 ):
        
        
        self.na_threshold = na_threshold
        self.cardinality_threshold = cardinality_threshold
        self.drop_high_na_cols = drop_high_na_cols
        self.drop_zero_variance_features = drop_zero_variance_features
        self.drop_high_cardinal_features = drop_high_cardinal_features
        self.encoding = encoding
        self.memory_reduction = memory_reduction
        self.export = export
        

    def type_assigner(self, data):
        
        assert type(data) == pd.DataFrame, "Provide a pandas dataframe for type assignment."
        
        print("---------------------- TYPE ASSIGNMENT STARTED -----------------------------")
        
        object_cols = data.select_dtypes(include = "object").columns.tolist()
        
        numerical_cols = data.select_dtypes(include = "number").columns.tolist()
        
        date_cols = data.select_dtypes(exclude = ["object", "number"]).columns.tolist()
        
        high_cardinal_cols = []
        low_mid_cardinal_cols = []
        
        for col in data[object_cols].columns:
            
            try:
                
                data[col] = data[col].astype(float)
                numerical_cols.append(col)
                object_cols.remove(col)
            
            except:
                
                try:
                    data[col] = pd.to_datetime(data[col], format = "%d.%m.%Y")
                    date_cols.append(col)
                    object_cols.remove(col)
                    
                except:
                    
                    if data[col].nunique() >= self.cardinality_threshold:
                        
                        high_cardinal_cols.append(col)
                        
                    else:
                        
                        low_mid_cardinal_cols.append(col)
                        
        self.numerical_cols = numerical_cols
        self.high_cardinal_cols = high_cardinal_cols
        self.low_mid_cardinal_cols = low_mid_cardinal_cols
        self.date_cols = date_cols

        
        
        print(f"Number of numerical columns is : {len(self.numerical_cols)}")
        
        print(f"Number of high cardinality columns is : {len(self.high_cardinal_cols)}")
        
        print(f"Number of low_mid cardinality columns is : {len(self.low_mid_cardinal_cols)}")
        
        print(f"Number of date columns is : {len(self.date_cols)}")
        
        print("---------------------- TYPE ASSIGNMENT FINISHED -----------------------------")
       
        print("-----------------------------------------------------------------------------", "\n")
    
    
    def missing_control(self, data):
        
        print("------------------- HIGH NA FEATURE DETECTION STARTED --------------------------")
        
        high_na_cols = []
        
        for col in data.columns:
            
            if data[col].isnull().mean() >= self.na_threshold:
                
                high_na_cols.append(col)
                
        if self.drop_high_cardinal_features:
            
            self.high_na_cols = [col for col in high_na_cols if col not in self.high_cardinal_cols] 
            
        else:
            
            self.high_na_cols = high_na_cols
        
        print(f"Detected high Missing Ratio cols. Total number of high missing ratio columns is : {len(self.high_na_cols)}." )
        
        print("-----------------------------------------------------------------------------", "\n")
    
    
    def zero_variance_detector(self, data):
        
        print("------------------- ZERO-VARIANCE FEATURE DETECTION STARTED --------------------------")
        
        zero_variance_cols = []
        
        for col in data.columns:
            
            if data[col].nunique() == 1:
                
                zero_variance_cols.append(col)
        
        if self.drop_high_na_cols:
        
            self.zero_variance_cols = [col for col in zero_variance_cols if col not in self.high_na_cols]
            
        else:
            
            self.zero_variance_cols = zero_variance_cols
        
        print(f"Detected zero-variance columns. Total number of zero-variance columns is : {len(self.zero_variance_cols)}.")
        
        print("-----------------------------------------------------------------------------", "\n")
        
        
    def encoder(self, X, y):
        
        print("------------------- ENCODING MAP CREATION STARTED --------------------------")
        
        X2 = X.copy()
        
        X2["target"] = y.values
        
        mapping_dict = {}

        for col in self.low_mid_cardinal_cols: 
            
            assert col in X2, f"{col} is not included in given dataframe. Please check."
    
            mapping_dict.update( { col : X2.groupby(col)["target"].mean().to_dict() } )
        
        self.mapping_dict = mapping_dict
        
        print("------------------- ENCODING MAP CREATION FINISHED --------------------------")
        
        print("-----------------------------------------------------------------------------", "\n")
    
    
    def reduce_mem_usages(self, df):
        
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        
        #start_mem = df.memory_usage().sum() / 1024**3
        #print('Memory usage of dataframe is {:.2f} GB'.format(start_mem))
        
        for col in df.columns:
            
            col_type = df[col].dtype
            
            if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
                
                c_min = df[col].min()
                
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        
                        df[col] = df[col].astype(np.int8)
                        
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        
                        df[col] = df[col].astype(np.int16)
                        
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        
                        df[col] = df[col].astype(np.int32)
                        
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        
                        df[col] = df[col].astype(np.int64)  
                else:
                    
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        
                        df[col] = df[col].astype(np.float16)
                        
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        
                        df[col] = df[col].astype(np.float32)
                    else:
                        
                        df[col] = df[col].astype(np.float64)
                        
            #elif 'datetime' not in col_type.name:
                #df[col] = df[col].astype('category')

        #end_mem = df.memory_usage().sum() / 1024**3
        #print('Memory usage after optimization is: {:.2f} GB'.format(end_mem))
        #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
       
        
        return None
    
    
    def export_files(self):
        
        print(" $$$ EXPORTING NECESSARY FILES TO OUTPUTS DIRECTORY $$$")
            
        now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            
        if os.path.exists("./outputs") == False:
            
            os.mkdir("./outputs")
            
            os.mkdir(f"./outputs/{now}")
            
        else:
                
            os.mkdir(f"./outputs/{now}")
            
        joblib.dump(self.numerical_cols, f"./outputs/{now}/numerical_columns.joblib")
        
        joblib.dump(self.low_mid_cardinal_cols, f"./outputs/{now}/low_mid_cardinal_cols.joblib")
        
        joblib.dump(self.date_cols, f"./outputs/{now}/date_columns.joblib")
            
        joblib.dump(self.high_cardinal_cols, f"./outputs/{now}/high_cardinal_columns.joblib")
        
        joblib.dump(self.high_na_cols, f"./outputs/{now}/high_na_columns.joblib")
        
        joblib.dump(self.zero_variance_cols, f"./outputs/{now}/zero_variance_columns.joblib")
        
        if self.encoding:
            
            joblib.dump(self.mapping_dict, f"./outputs/{now}/mapping_dict.joblib")
            
            
        print("$$$ EXPORTING FINISHED - PLEASE CHECK OUTPUTS DIRECTORY  $$$", "\n")
        
        
    
    
    def fit(self, X, y):
        
        print("**********************  FITTING DATA TO PREPROCESS METHODS STARTED **********************")
        
        self.type_assigner(X)
        
        self.missing_control(X)
        
        self.zero_variance_detector(X)
        
        if self.encoding:
            
            self.encoder(X,y)
        
        print("**********************  FITTING DATA TO PREPROCESS METHODS FINISHED **********************", "\n")
        
        
        if self.export:
            
            self.export_files()
                
        return self
      
        
    def transform(self, X):
        
        print("//////////////////////////  TRANSFORMATION STARTED  //////////////////////////")
        
        X2 = X.copy()
        
        if self.memory_reduction:
            
            print("----- MEMORY REDUCTION STARTED -----")
            
            self.reduce_mem_usages(X2)
            
            print("----- MEMORY REDUCTION FINISHED -----" , "\n")
            
        else:
            pass
        
        
        if self.drop_high_cardinal_features:
            
            print("----- dropping high cardinal cols -----")
            
            X2.drop(self.high_cardinal_cols, axis = 1, inplace = True)
        
        
        if self.drop_high_na_cols:
            
            print("----- dropping high NA cols -----")
            
            X2.drop(self.high_na_cols, axis = 1, inplace = True)
            
            
        if self.drop_zero_variance_features:
            
            print("----- dropping zero variance cols -----")
            
            X2.drop(self.zero_variance_cols, axis = 1, inplace = True)
            
            
        if self.encoding:
            
            print("Encoding started")
            
            encode_cols = [col for col in X2.columns if col in self.mapping_dict.keys()]
            
            for col in encode_cols:
                
                # new label control, if new label exists, map with np.nan
                
                labels = X2[col].dropna().unique().tolist()
                
                for label in labels:
                    
                    if label not in self.mapping_dict[col].keys():
                        
                        self.mapping_dict[col].update( { label : np.nan } )
                        
                X2[col] = X2[col].map(self.mapping_dict[col])
                
            print("Encoding finished", "\n")
                
        
        print("//////////////////////////  TRANSFORMATION FINISHED  //////////////////////////")
        
            
        return X2
               
        
    def fit_transform(self, X, y):
        
        self.fit(X,y)
        
        df_transformed = self.transform(X)
        
        return df_transformed
        
        
def combine_list(mylist):
    
    final_list = []
    
    for list_ in mylist:
        
        for element in list_:
            
            final_list.append(element)
    
    return final_list


def data_processing(processor,
                    path_to_data,
                    index_col,
                    date_col,
                    target_col,
                    date_format="%d.%m.%Y",
                    cut_off_date="2022-03-01"):
    
    current_path = os.getcwd()

    os.chdir(path_to_data)

    files = glob.glob("*.csv")

    numerical_cols = []

    low_mid_cardinal_cols = []
    
    high_cardinal_cols = []

    date_cols = []

    high_na_cols = []

    zero_variance_cols = []

    mapping_dict = {}

    train_data = []

    test_data = []

    for filename in files:

        print(filename, "\n")

        data = pd.read_csv(filename,
                           delimiter=";",
                           header=0,
                           encoding="ISO-8859-1",
                           low_memory=False,
                           decimal=',')

        assert index_col in data, "index_col is not in the data."
        assert date_col in data, "date_col is not in the data."
        assert target_col in data, "target_col is not in the data"

        data.set_index(index_col, inplace=True)

        data[date_col] = pd.to_datetime(data[date_col], format=date_format)

        train = data[data[date_col] <= cut_off_date]

        test = data[data[date_col] > cut_off_date]

        X_train = train.drop([date_col, target_col], axis=1)

        y_train = train[target_col]

        X_test = test.drop([date_col, target_col], axis=1)

        y_test = test[target_col]

        processor.fit(X_train, y_train)

        X_train_transformed = processor.transform(X_train)

        X_test_transformed = processor.transform(X_test)
        
        remained_object_cols = X_test_transformed.select_dtypes(exclude = "number").columns.tolist()
        
        if remained_object_cols:
            
            for col in remained_object_cols:
                
                if col in processor.numerical_cols:
                    
                    X_test_transformed[col] = X_test_transformed[col].astype(float)
                
                elif col in processor.date_cols:
                    
                    X_test_transformed[col] = pd.to_datetime(X_test_transformed[col], format = "%d.%m.%Y")
                
                else:
                    
                    print(f"{col.upper()} Could Not Converted to numeric. Further processing required...")
        
        del data
        del train
        del test
        del X_train
        del X_test

        numerical_cols.append(processor.numerical_cols)

        low_mid_cardinal_cols.append(processor.low_mid_cardinal_cols)

        date_cols.append(processor.date_cols)

        high_cardinal_cols.append(processor.high_cardinal_cols)

        high_na_cols.append(processor.high_na_cols)

        zero_variance_cols.append(processor.zero_variance_cols)

        mapping_dict.update(processor.mapping_dict)

        train_data.append(X_train_transformed)

        test_data.append(X_test_transformed)
        
        print(filename, "is processed and appended.", "\n")
        
        
    os.chdir(current_path)

    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    if os.path.exists("./outputs") == False:

        os.mkdir("./outputs")

        os.mkdir(f"./outputs/{now}")

    else:

        os.mkdir(f"./outputs/{now}")
        
        
    numerical_cols = combine_list(numerical_cols)
    
    low_mid_cardinal_cols = combine_list(low_mid_cardinal_cols)
    
    date_cols = combine_list(date_cols)
    
    high_cardinal_cols = combine_list(high_cardinal_cols)
    
    high_na_cols = combine_list(high_na_cols)
    
    zero_variance_cols = combine_list(zero_variance_cols)

    joblib.dump(numerical_cols, f"./outputs/{now}/numerical_columns.joblib")

    joblib.dump(low_mid_cardinal_cols,
                f"./outputs/{now}/low_mid_cardinal_cols.joblib")

    joblib.dump(date_cols, f"./outputs/{now}/date_columns.joblib")

    joblib.dump(high_cardinal_cols,
                f"./outputs/{now}/high_cardinal_columns.joblib")

    joblib.dump(high_na_cols, f"./outputs/{now}/high_na_columns.joblib")

    joblib.dump(zero_variance_cols,
                f"./outputs/{now}/zero_variance_columns.joblib")

    joblib.dump(mapping_dict, f"./outputs/{now}/mapping_dict.joblib")

    train_df = pd.concat(train_data, axis=1, join="inner")

    test_df = pd.concat(test_data, axis=1, join="inner")
    
    print(train_df.shape, y_train.shape, test_df.shape, y_test.shape)
    
    train_final = pd.concat([train_df, y_train], axis = 1, join = "inner")
    
    test_final = pd.concat([test_df, y_test], axis = 1, join = "inner")

    print(train_final.shape, test_final.shape)

    return train_final, test_final