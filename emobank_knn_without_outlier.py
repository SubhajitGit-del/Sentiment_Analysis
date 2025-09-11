
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
#df=emo_vector.drop(['id','text','split'],axis=1)
dataset.info()



emo_vector = pd.get_dummies(emo_vector, columns = ['text_polarity'])


#emo_vector=emo_vector.drop(['Dominance','Arousal'],axis=1)
emo_vector.head()

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






import math
import operator
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error



def cosine_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    #print(cos_sim)
    return abs(cos_sim)

def knn(train_data,train_V, testInstance, k):
    distances = []
    sort = []
    length = len(testInstance)
    
    
     # Calculating euclidean distance / cosine similerity between each row of training data and test data
    for x in range(len(train_data)):
        dist = cosine_similarity(testInstance, train_data[x])
        distances.append((dist,x))
     
  
     # Sorting them on the basis of cosine similerity in descending order
    sorted_d = sorted(distances,key=lambda l:l[0],reverse = True)
     
    # print(sorted_d)
    avg=0
    for i in range(0,k):
         avg+= train_V[sorted_d[i][1]]
    return avg/k

def calPercent(x, y):
   percent = x * y/100
   return math.ceil(percent)
    



lower = 0
upper =0
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


    
    k_val=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    group_accuracy =[]
    for k in range(0,len(k_val)):
        if len(train_data)<k_val[k]:
            group_accuracy.append(0)
        else:
            predicted_val=[]
            for j in range(0,len(test_data)):
                print(j)
                predicted_val.append(knn(train_data,train_V,test_data[j],k_val[k]))
            
            
            sum = 0
            for i in range(0,len(test_V)):
                sum += (abs(test_V[i]-predicted_val[i])/test_V[i])
            APEe = sum / len(test_V)
            
            # Computing the absolute percent error
            APEe=100-(100*APEe)
            print('The Accuracy of KNN model is:', APEe,'->',k_val[k])
            group_accuracy.append(APEe)
    
    Global_accuracy.append(group_accuracy)
    
group_mean =[]
group_mean.append(np.mean(Global_accuracy,axis=0)) 

    
print(group_mean)  






