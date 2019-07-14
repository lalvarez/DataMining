# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:29:28 2019

@author: Laura
"""

#Read data file
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import string
from nltk.corpus import stopwords
from scipy.spatial import distance

#Number of neighbors
k = 25

#Ecuclidean distance of two vectors
def distanciaeuclidea(x1, x2, y1, y2):
    dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    dist = round(dist,5)
    return dist

#Calculates the mist repeated label in the k neighbors set of labels
def decision(closest_labels):
    contador = 0
    for label in closest_labels:
        if label == -1:
            contador -= 1
        elif label == +1:
            contador += 1
    if contador > 0:
        flabel= '+1'
        print(i,flabel)
        f.write(flabel)
        f.write("\n")
    else:
        flabel= '-1'
        print(i,flabel)
        f.write(flabel)
        f.write("\n")

#Calculates the K closests points 
def find_closest(point):
    closest = []
    for t in model_data:
        dist = distanciaeuclidea(point[0],t[0][0],point[1],t[0][1])
        closest.append((dist, t[1])) 

    closest.sort(key=lambda tup: tup[0])
    kclosest = closest[:k]
    closest_labels = [c[1] for c in kclosest]
    decision(closest_labels)

#Cleans the punctuation and other meaningless characters from the reviews
def remove_punctuation(review):
    review = ''.join(ch for ch in review if ch not in exclude)
    return review

#First we read the training data an save it into a pandas dataframe
df = pd.read_fwf('1548889051_0353532_train.dat', header=None, widths=[2, int(1e5)], names=['label', 'text'])

#We do the same with our test data
testdf = pd.read_fwf('1548889052_1314285_test.dat', header=None, engine='python', widths=[int(1e5)], names=['text'])
#We open our file where we are going to write our results
f = open("FinalResult3.dat", "w+")

#Structures needed to manage data
model_data = []
model_X = []
model_Y = []
test_X = []
test_Y = []
closest = []


exclude = set(string.punctuation)


#First we create out list with the words we should ignore
s = ''
adj=[]
listaStop = list(set(stopwords.words('english')))
stop_words = ['/>the','.',',','(',')','~','@','#','$','%','^','&','*','-','+','=',':',';','"','<','>','/','?','/><br']
for ch in stop_words:
    listaStop.append(ch)
#We have in listaStop all the stopwords to ignore

#Now, we create another list with the positive adjetives that appear the most on our data
positive = ['simple','different','thumbs','adore','well','loved','good','recommend','excelent','amazing','classic','french','screen','perfect','romantic','greatest','social','love','touching','deep','enjoyable','fantastic','sexual','perfect','favorite','expecting','hilarious','modern','actual','decent','enjoyable','powerful','happy','emotional','wonderful','nice','beautiful','better','excellent','funny','original','real','best','great','good','realistic','visual','clear','happy','interesting','complete','brilliant','nice']

#We are also going to have another list with the negative adjeives
negative = ['worse','ridiculous','skip','stop','crap','mistake','disgusting','bad','sucks','problem','unfortunately','disappointed','bother','difficult','annoyed','garbage','special','entire','stupid','ridiculous','bored','typical','poor','awful','low','worse','ridiculous','horrible','cheap','difficult','dramatic','long','poor','weak','worst','slow','bad']

#Auxiliar analyzer 
sid = SentimentIntensityAnalyzer()

#In this loop we calculate the polarity the leng and the meaning words
for i, row in df.iterrows():
    review = row['text']
    #Calculate the polarity
    ss = sid.polarity_scores(review)
    #Round polarity to 3 digits
    c = round(ss['compound'],2)
    pos = 0
    neg = 0
    #Analise some adjetives from the text
    words = row['text'].split()
    reviewfiltrada = [word for word in words if word not in listaStop]
    for wordto in reviewfiltrada:
        if wordto in positive:
            pos += 1
        if wordto in negative:
            neg += 1
    word_value = pos - neg
    model_X.append(c)
    model_Y.append(word_value)
    point = (c, word_value )
    model_data.append((point,row['label'] ))

plt.figure(figsize=(50,20))
plt.scatter(model_X,model_Y,alpha=0.5, color='red')
plt.autoscale()
plt.show()

for i, row in testdf.iterrows():
    review = row['text']
    ss = sid.polarity_scores(review)
    c = round(ss['compound'],2)
    pos = 0
    neg = 0
    words = row['text'].split()
    reviewfiltrada = [word for word in words if word not in listaStop]
    for wordto in reviewfiltrada:
        if wordto in positive:
            pos += 1
        if wordto in negative:
            neg += 1
    word_value = pos - neg
    if(i ==0 or i == 20 or i==200 or i ==340 or i== 500 or i ==1000 or i ==2304 or i ==3444 or i ==9833):
        test_X.append(c)
        test_Y.append(word_value)
    point = (c, word_value )
    find_closest(point)

f.close()

plt.figure(figsize=(50,20))
plt.scatter(model_X,model_Y,color='red')
plt.scatter(test_X,test_Y,color='blue')
plt.show()