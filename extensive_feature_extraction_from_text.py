# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 11:02:18 2023

@author: subha
"""

import pandas as pd
import numpy as np
import string 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df=pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/survey_data_rowwise.csv")
#df = pd.read_csv("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/new dataset/iemocapTrans.csv")

#df=df.drop(['Count_sentiWord','SentiScore'],axis=1)
#df=df.drop(['end_time','start_time','titre','translated'],axis=1)
df=df.drop(['index'],axis=1)



# for survey dataset
for index in df.index:
    df['text'][index]=str(df['text'][index])


# text length including spaces
df["char_count"] = df["text"].apply(len)

df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# average length of the words used in the essay
df['word_density'] = df['char_count'] / (df['word_count']+1)


# set of punctuation ----> !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# count stopwords in each text
stopWords = set(stopwords.words('english'))
count_array =[]
for index in df.index:
    data = df['text'][index]
    words = word_tokenize(data)
    wordsFiltered = []
    
    for w in words:
        if w in stopWords:
            wordsFiltered.append(w)        
    count_array.append(len(wordsFiltered)) 
    
df['stopwords_count']  = count_array


# distinct words in each text
unique_word_count =[]
repeating_word_count =[]
for index in df.index:
    data = df['text'][index]
    total_words = data.split(' ')
    unique_words = set(data.split(' '))
    dist_word_count = len(unique_words)
    non_dist_word_count = len(total_words) - dist_word_count
    unique_word_count.append(dist_word_count)
    repeating_word_count.append(non_dist_word_count)
df['unique_words_count']  = unique_word_count
df['repeating_word_count']  = repeating_word_count



# average word lenth calculation
avg_len =[]
for index in df.index:
    data = df['text'][index]
    words = data.split(' ')
    
    if total_words ==0:
        avg_len.append(0)
        continue
    length_words = 0
    for w in range(0,len(words)):
        length_words += len(words[w])
    avg_len.append(float(length_words)/len(words))    

df['avg_word_length'] = avg_len



# parts of speech count 
# JJ-->Adjective, VB-->Verb, RB-->Adverb,NNS-->Noun,singular form

Adjective =[]
Verb =[]
Adverb =[]
Noun =[]
from collections import Counter
for index in df.index:
    data = df['text'][index]
    tokens = nltk.word_tokenize(data.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags)
    #print(counts['JJ'],counts['VB'],counts['RB'],counts['NNS'])
    Adjective.append(counts['JJ'])
    Verb.append(counts['VB'])
    Adverb.append(counts['RB'])
    Noun.append(counts['NNS'])
    
df['Adjctive_count'] = Adjective
df['Verb_count'] =Verb
df['Adverb_count'] = Adverb
df['Noun_count'] = Noun


def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

subjectivity =[]
for index in df.index:
    data = df['text'][index]
    subjectivity.append(getSubjectivity(data))

df['subjectivity_text']=subjectivity



def sentiment_scores_sentence(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        sentiment = "Positive"
        
 
    elif sentiment_dict['compound'] <= - 0.05 :
        sentiment = "Negetive"
    
    else :
        sentiment ="Neutral"
        
    return  sentiment_dict['neg'],sentiment_dict['neu'],sentiment_dict['pos'],sentiment_dict['compound'],sentiment
        
negative=[]
positive=[]
neutral =[]
compound =[]
polarity =[]
for index in df.index:
    data = df['text'][index]
    neg,neu,pos,comp,senti = sentiment_scores_sentence(data)
    negative.append(neg)
    positive.append(pos)
    neutral.append(neu)
    compound.append(comp)
    polarity.append(senti)
df['text_polarity']=polarity
df['text_sentiment_score']=compound
df['text_neg_score']=negative
df['text_pos_score']=positive
df['text_neu_score']=neutral   


#print(ssentiment_scores('good')) 


# sentiscore of a word
def sentiment_scores_word(word):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(word)
    
    max_val = max(sentiment_dict['neg'],sentiment_dict['pos'],sentiment_dict['neu'])
    
    #print(max_val,sentiment_dict['neg'],sentiment_dict['pos'],sentiment_dict['neu'] )
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['pos'] == max_val :
        sentiment = "Positive"
        
 
    elif sentiment_dict['neg'] == max_val :
        sentiment = "Negetive"
    
    else :
        sentiment ="Neutral"
    
    #print(word,sentiment_dict['compound'])
    return  sentiment




neg_words_count=[]
pos_words_count=[]
neu_words_count=[]
for index in df.index:
    print(index)
    data = df['text'][index]
    words = data.split(' ')
    neu_count,pos_count,neg_count =0,0,0
    #neu_score,pos_score,neg_score =0,0,0
    for i in range(0,len(words)):
        senti = sentiment_scores_word(words[i])
        
        #print(score,senti)
        if senti == "Positive":
            pos_count+=1
            #pos_score+=score
            
        elif senti == "Negetive":
            neg_count+=1
            #neg_score+=score
        else:
            neu_count+=1
            #neu_score+=score
            
    neg_words_count.append(neg_count)
    pos_words_count.append(pos_count)
    neu_words_count.append(neu_count)
    
df['neg_words_in_text']=neg_words_count
df['pos_words_in_text']=pos_words_count
df['neu_words_in_text']=neu_words_count




df.to_excel("C:/Users/subha/OneDrive/Desktop/ml_6th_sem_project/New folder/surveyData_custom_features.xlsx",index=False)




        