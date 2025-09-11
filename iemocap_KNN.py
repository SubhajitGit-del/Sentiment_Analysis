# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:48:00 2023

@author: subha
"""


import pandas as pd
import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split   


imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/Emocap_vector_CSV.csv")
#imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/extensive_features_extraction_imocap_csv.csv",encoding='latin1')
#imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/iemocap_vector_bert_plus_custom_csv.csv")

imocap_group = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/iemocap_groups_csv.csv")
imocap_group=imocap_group.to_numpy()

    


dataset=imocap_dataset.drop(['id_no','Dominance','Arousal'],axis=1)



from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    #print(cos_sim)
    return abs(cos_sim)

def knn(X_train,y_train,testInstance,k):
    distances = []
    

    # Calculating cosine similerity between each row of training data and test data
    for x in range(len(X_train)):
        dist = cosine_similarity(testInstance, X_train[x])

        distances.append((dist, x))

    # Sorting them on the basis of cosine similerity in descending order
    sorted_d = sorted(distances, key=lambda l: l[0], reverse=True)

    avg = 0
    for i in range(0, k):
        avg += y_train[sorted_d[i][1]]
    return avg / k

accuracy_diff_r=[]
r_state = [8]
for rr in range(0,len(r_state)):
    globa_accuracy_for_diff_K =[]
    for grp in range(0,len(imocap_group)):
        lower = imocap_group[grp][1]-2
        upper = imocap_group[grp][2]-1
        

        group_data = []
        for i in range(lower,upper):
                group_data.append(dataset.iloc[i])    
        group_data=pd.DataFrame(group_data)
        #group_data = pd.get_dummies(group_data, columns = ['text_polarity'])
    
        X = group_data.drop('Valence', axis=1)
        y =group_data.Valence
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r_state[rr])
    
        X_train= np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
    
    
        accuracy =[]  # group_accuracy_for_diff_K
        k_val = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        for k in range(0,len(k_val)):
            if len(X_train)<k_val[k]:
                accuracy.append(0)
            else:
                predicted_val=[]
                for i in range(0,len(X_test)):
                    predicted_val.append(knn(X_train,y_train,X_test[i],k_val[k]))
             
                sum = 0
                for i in range(0,len(y_test)):
                    sum += (abs(y_test[i]-predicted_val[i])/y_test[i])
                APEe = sum / len(y_test)
                
                # Computing the absolute percent error
                APEe=100 *(1-APEe)
                accuracy.append(APEe)
        globa_accuracy_for_diff_K.append(accuracy)  
    
    group_mean =[]
    group_mean.append(np.mean(globa_accuracy_for_diff_K,axis=0))   
    accuracy_diff_r.append(group_mean)
    
print(accuracy_diff_r)
 
        



        