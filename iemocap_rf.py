

# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:48:00 2023

@author: subha
"""


import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/Emocap_vector_CSV.csv")
#imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/extensive_features_extraction_imocap_csv.csv",encoding='latin1')
imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/iemocap_vector_bert_plus_custom_csv.csv")

imocap_group = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/iemocap_groups_csv.csv")
imocap_group=imocap_group.to_numpy()


dataset=imocap_dataset.drop(['id_no','Arousal','Valence'],axis=1)




accuracy_diff_r=[]
r_state = [8]
for rr in range(0,len(r_state)):
    accuracy =[] 
    
    for grp in range(0,len(imocap_group)):
        lower = imocap_group[grp][1]-2
        upper = imocap_group[grp][2]-1
        
        
        group_data = []
        for i in range(lower,upper):
                group_data.append(dataset.iloc[i])    
        group_data=pd.DataFrame(group_data)
        group_data = pd.get_dummies(group_data, columns = ['text_polarity'])
    
        X = group_data.drop('Dominance', axis=1)
        y =group_data.Dominance
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r_state[rr])
        
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test =scaler.transform(X_test)
    
    
    
        X_train= np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
    
    
         # group_accuracy_for_diff_K
        rf = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, y_train)
        predicted_val = rf.predict(X_test)
             
        sum = 0
        for i in range(0,len(y_test)):
            sum += (abs(y_test[i]-predicted_val[i])/y_test[i])
        APEe = sum / len(y_test)
        
        # Computing the absolute percent error
        APEe=100 *(1-APEe)
        accuracy.append(APEe)
    
    mean =[]
    mean.append(np.mean(accuracy,axis=0)) 
    accuracy_diff_r.append(mean)

print(accuracy_diff_r)   
        




