# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:49:13 2023

@author: subha
"""

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


common_outliers= pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Outliers/EMOBANK_OUTLIERS/Valence/val_outliers_z_score.csv")
#common_outliers = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Outliers/EMOBANK_OUTLIERS/X_OR_EMOBANK_val_outliers.csv")

common_outliers_list = common_outliers['id_no'].to_list()
common_outliers_list.sort()

dict = {}
for i in range(0,len(common_outliers_list)):
    index = common_outliers_list[i]
    dict[index] = i 
    


dataset=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/valance_classification_dataset.csv")
emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Emobank_vectorr.csv")
#emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/extensive_feature_extraction_from_text_imobank_csv2.csv",encoding='latin1')
#emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/emobank_vector_bert_plus_custom_csv.csv",encoding='latin1')
dataset.info()


#emo_vector = pd.get_dummies(emo_vector, columns = ['text_polarity'])


# spliting entire emobank dataset into similer groups
ids = dataset['id']
split_ids = []
for i in range(0,len(ids)):
    split_id = ids[i].split('_')
    split_ids.append(split_id)


#count the no. of such groups
group_counts = []
count = 1
for i in range(0,len(split_ids)-1):
    if split_ids[i][0]==split_ids[i+1][0]:
        count += 1
    else:
        group_counts.append(count)
        count = 1
group_counts.append(count)




lower = 0
upper =0

group_accurecy_dom=[]
group_accurecy_aro=[]
group_accurecy_val=[]
group_mean=[]




emo_val = 'Valance'
for g in range(0,len(group_counts)):
    lower=lower+upper        
    upper = group_counts[g]
    train_data=[]
    train_V=[]
    test_data=[]
    test_V=[]
    
   
    for i in range(lower,lower+upper):
        if dict.get(i) is None:
            if(dataset.iloc[i]["split"]=="train"):
                train_data.append(emo_vector.iloc[i])
                train_V.append(dataset.iloc[i][emo_val])
            if(dataset.iloc[i]["split"]=="test" or dataset.iloc[i]["split"]=="dev" ):
                test_data.append(emo_vector.iloc[i])
                test_V.append(dataset.iloc[i][emo_val]) 
        
                
    
    if len(test_V) == 0:
        continue
    
    
# =============================================================================
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#     test_data =scaler.transform(test_data)
# =============================================================================

    
    X_train = np.asarray(train_data)
    y_train = np.asarray(train_V)
    X_test = np.asarray(test_data)
    y_test = np.asarray(test_V)
    

    rf = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    
    sum = 0
    for i in range(0,len(y_test)):
        sum += (abs(y_test[i]-predictions[i])/y_test[i])
    APEe = sum / len(y_test)

    # Computing the absolute percent error
    APEe=100*(1-APEe)
    print('The Accuracy of Rf model is:', APEe)
    group_accurecy_val.append(APEe)
    
    
            
 
group_mean.append(np.mean(group_accurecy_val))
print(np.mean(group_accurecy_val))
                 
print(group_mean)