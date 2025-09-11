# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:51:13 2023

@author: subha
"""

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


common_outliers= pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Outliers/EMOBANK_OUTLIERS/Valence/val_outliers_z_score.csv")


common_outliers_list = common_outliers['id_no'].to_list()
common_outliers_list.sort()

dict = {}
for i in range(0,len(common_outliers_list)):
    index = common_outliers_list[i]
    dict[index] = i 
    






dataset=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/valance_classification_dataset.csv")
emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Emobank_vectorr.csv")
dataset.info()





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
#print(group_counts)




lower = 0
upper =0

group_accurecy_dom=[]
group_accurecy_aro=[]
group_accurecy_val=[]
group_mean=[]

positive_outlier=[]
negetive_outlier=[]
Global_outlier_predicion=[]
Global_y_outlier=[]

pos_y_outlier =[]
neg_y_outlier =[]

emo_val = 'Dominance'
for g in range(0,len(group_counts)):
    lower=lower+upper        
    upper = group_counts[g]
    train_data=[]
    train_V=[]
    test_data=[]
    test_V=[]
    
    X_outlier =[]
    y_outlier =[]
    
    upper_limit=  4.0229
    lower_limit= 1.9305
    
    check_outLier=[]
    for i in range(lower,lower+upper):
        if dict.get(i) is None:
            if(dataset.iloc[i]["split"]=="train"):
                train_data.append(emo_vector.iloc[i])
                train_V.append(dataset.iloc[i][emo_val])
            if(dataset.iloc[i]["split"]=="test" or dataset.iloc[i]["split"]=="dev" ):
                test_data.append(emo_vector.iloc[i])
                test_V.append(dataset.iloc[i][emo_val]) 
        else:
            X_outlier.append(emo_vector.iloc[i])
            y_outlier.append(dataset.iloc[i][emo_val])
            if dataset.iloc[i][emo_val] < lower_limit:
                neg_y_outlier.append(dataset.iloc[i][emo_val])
            elif dataset.iloc[i][emo_val] >  upper_limit:
                pos_y_outlier.append(dataset.iloc[i][emo_val])
                
    
    if len(test_V) == 0:
        continue
    
    


    
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
    
    if len(X_outlier)!=0:
        outlier_predictions = rf.predict(X_outlier)
        Global_outlier_predicion.append(outlier_predictions)
        Global_y_outlier.append(y_outlier)
        
        for j in range(0,len(outlier_predictions)):
            if outlier_predictions[j]< lower_limit:
                negetive_outlier.append("negetive")
            elif outlier_predictions[j]> upper_limit:
                positive_outlier.append("positive")
            
 
group_mean.append(np.mean(group_accurecy_val))
print(np.mean(group_accurecy_val))
                 
print(group_mean)




l = len(Global_outlier_predicion)
for i in range(0,l):
    l1 = len(Global_outlier_predicion[i])
    for j in range(0,l1):
        print(Global_y_outlier[i][j],' ->',Global_outlier_predicion[i][j])

print(len(positive_outlier)+len(negetive_outlier)," ->",len(common_outliers_list))


print("actual-->"," pos ",len(pos_y_outlier),"neg ",len(neg_y_outlier))
print("predict-->"," pos ",len(positive_outlier),"neg ",len(negetive_outlier))
