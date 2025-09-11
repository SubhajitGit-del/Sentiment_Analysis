# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:51:13 2023

@author: subha
"""

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler


common_outliers = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Outliers/EMOBANK_OUTLIERS/X_OR_EMOBANK_dom_outliers.csv")
common_outliers_list = common_outliers['id_no_dom'].to_list()
common_outliers_list.sort()

dict = {}
for i in range(0,len(common_outliers_list)):
    index = common_outliers_list[i]
    dict[index] = i  


dataset=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/valance_classification_dataset.csv")
#emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/Emobank_vectorr.csv")
#emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/extensive_feature_extraction_from_text_imobank_csv2.csv",encoding='latin1')
emo_vector=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/emobank_vector_bert_plus_custom_csv.csv",encoding='latin1')
dataset.info()


emo_vector = pd.get_dummies(emo_vector, columns = ['text_polarity'])




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
                    hp.Int('units' + str(i),min_value=8,max_value=128,step=8),
                    activation=hp.Choice('activation'+ str(i),values=['relu','tanh','sigmoid'])
                    )
                )
            
        counter+=1 
        
    model.add(Dense(1)) 
    
    
    model.compile(optimizer=hp.Choice('optimizer',values=['rmsprop','adam','sgd']),
                 loss=hp.Choice('loss_function',values=['mean_squared_error','mean_absolute_error']))
    
    return model




lower = 0
upper =0
Global_accuracy_val =[] 
Global_accuracy_aro =[] 
Global_accuracy_dom =[] 

Global_accuracy =[]
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
                train_V.append(dataset.iloc[i]['Dominance'])
            if(dataset.iloc[i]["split"]=="test" or dataset.iloc[i]["split"]=="dev" ):
                test_data.append(emo_vector.iloc[i])
                test_V.append(dataset.iloc[i]['Dominance']) 
    
    if len(test_V) == 0:
        continue
    
    
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data =scaler.transform(test_data)

    
    train_x = np.asarray(train_data)
    train_y = np.asarray(train_V)
    test_x = np.asarray(test_data)
    test_y = np.asarray(test_V)
    
    
    
    
    # Initialize a tuner (here, RandomSearch). We use objective to specify the objective to select the best models,
    # and we use max_trials to specify the number of different models to try.

    tuner = keras_tuner.RandomSearch(build_model,objective='val_loss',max_trials=5,directory="mydir",project_name='final_pro_grpWise_Excluding OUTLIERs_BERT+Custom_emo_d')
    tuner.search(train_x,train_y, epochs=100)  


    # Start the search and get the best model:
    #tuner.get_best_hyperparameters()[0].values

    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.fit(x=train_x, y=train_y, epochs=100, batch_size=32 )   

    # Errors in Train And Test data set
    best_model.evaluate(train_x,train_y)
    best_model.evaluate(test_x,test_y)

    predictions = best_model.predict(test_x)
    predictions_df = pd.DataFrame(np.ravel(predictions),columns=["Predictions"])
    comparison_df = pd.concat([pd.DataFrame(test_y,columns=["Real Values"]), predictions_df],axis=1)
    comparison_df
                 

    sum = 0
    for i in range(0,len(test_y)):
        sum += (abs(test_y[i]-predictions[i])/test_y[i])
    APEe = sum / len(test_y)

    # Computing the absolute percent error
    APEe =100- (100*APEe)
    print('The Accuracy of ANN model is:', APEe)
    Global_accuracy.append(APEe)

group_mean =[]
group_mean.append(np.mean(Global_accuracy,axis=0)) 

    
print(group_mean)
   
