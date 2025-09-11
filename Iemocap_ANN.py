# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:22:12 2023

@author: subha
"""

import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

#imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/Emocap_vector_CSV.csv")
#imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/extensive_features_extraction_imocap_csv.csv",encoding='latin1')
imocap_dataset = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/iemocap_vector_bert_plus_custom_csv.csv")

imocap_group = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/iemocap_groups_csv.csv")
imocap_group=imocap_group.to_numpy()

dataset=imocap_dataset.drop(['Valence','Arousal'],axis=1)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner
from tensorflow import keras


#Write a function that creates and returns a Keras model. Use the hp argument to define the hyperparameters 
#during model creation.
def build_model(hp):
    model = Sequential()
    
    counter = 0
    
    for i in range(hp.Int('num_layers', min_value=3,max_value=10)):
        
        if counter == 0:
            model.add(
                Dense(
                    hp.Int('units' + str(i),min_value=6,max_value=128,step=8),
                    activation=hp.Choice('activation'+ str(i),values=['relu','tanh','sigmoid']),
                    input_dim=792)
                )
        else :
            model.add(
                Dense(
                    hp.Int('units' + str(i),min_value=6,max_value=128,step=8),
                    activation=hp.Choice('activation'+ str(i),values=['relu','tanh','sigmoid'])
                    )
                )
            
        counter+=1 
        
    model.add(Dense(1)) 
    
    model.compile(optimizer=hp.Choice('optimizer',values=['rmsprop','adam','sgd']),
                 loss=hp.Choice('loss_function',values=['mean_squared_error','mean_absolute_error']))
    
    return model






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
        
        if len(group_data['text_polarity'].unique())<3:
            continue
        group_data = pd.get_dummies(group_data, columns = ['text_polarity'])
    
        X = group_data.drop('Dominance', axis=1)
        y =group_data.Dominance
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r_state[rr])
    
        X_train= np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        
        # overwrite=True
        tuner = keras_tuner.RandomSearch(build_model,objective='val_loss',max_trials=5,directory="mydir",project_name='final_pro_IEMOCAp_BERT+custom_ann_D')
        tuner.search(X_train,y_train, epochs=100)           
    
        # Start the search and get the best model:
        tuner.get_best_hyperparameters()[0].values
    
        best_model = tuner.get_best_models(num_models=1)[0]
    
        best_model.fit(x=X_train, y=y_train, epochs=100, batch_size=32) 
        predicted_val = best_model.predict(X_test)
             
        sum = 0
        for i in range(0,len(y_test)):
            sum += (abs(y_test[i]-predicted_val[i])/abs(y_test[i]))
        APEe = sum / len(y_test)
        
        # Computing the absolute percent error
        APEe=100 *(1-APEe)
        accuracy.append(APEe)
          
    
    mean =[]
    mean.append(np.mean(accuracy,axis=0)) 
    accuracy_diff_r.append(mean)

print(accuracy_diff_r)   


