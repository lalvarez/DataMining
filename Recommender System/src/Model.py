# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:41:26 2019

@author: Laura
"""
import pandas as pd
import numpy as np
from statistics import mean 

#1 LOAD THE TRAIN DATA INTO A DATAFRAME
df = pd.read_csv("train.dat",delimiter = ' ')
df = df.loc[:,:]
#TEST DATA: IDS TO PREDICT
test = pd.read_csv("test.dat",delimiter = ' ')

#Create matrix NxM with all 0s
n_users = df['userID'].nunique()
n_movies = df['movieID'].nunique()

matrix = np.full((n_users, n_movies), -1)

#AUX 
users = []
movies = []
last_user_i = 0
last_movie_i = 0
counter = 0

print("Creating creating matrix ...")
for i, row in df.iterrows():
    if(row['userID'] in users and row['movieID'] in movies):
            #Case user exists and movie too
            matrix[ users.index(row['userID']) , movies.index(row['movieID'])] = row['rating']
            counter += 1
    elif(row['userID'] in users):
        #Case user exists
        matrix[ users.index(row['userID']) , last_movie_i] = row['rating']
        movies.append(row['movieID'])
        last_movie_i += 1
        counter += 1
    elif(row['movieID'] in movies):
        #Case movie exists
        matrix[ last_user_i , movies.index(row['movieID'])] = row['rating']
        users.append(row['userID'])
        last_user_i += 1
        counter += 1
    else:
        #Case movie nor user don't exists
        matrix[last_user_i,last_movie_i] = row['rating']
        users.append(row['userID'])
        movies.append(row['movieID'])
        last_user_i += 1
        last_movie_i += 1
        counter += 1


index_i = 0
index_j = 0

print("Calculating initial values for unknown ratings ...")
#Unknown values we use the user mean of the movie
h = []
for i in matrix:
    #Calculate column mean: movie average rate
    for k in i:
        if(k != -1):
            h.append(k)
    m= mean(h)
    for j in i:
        if(j == -1):
            matrix[index_i,index_j]=m
        index_j += 1
    index_j = 0
    index_i += 1
    h = []

print("Calculating singular value descomposition of rating matrix ...")
u, s, vh = np.linalg.svd(matrix, full_matrices=True)

##Create the diagonal matrix
##We keep the 2 values
s_ = np.zeros((2113, 9936))
f = open("FinalResults.dat", "w+")
s_v = 12
for i in range(0,s_v):
    s_[i,i] = s[i]
#
print("Calculating reduction, number of singular values = ", s_v)
new_matrix = np.dot(np.dot(u,s_),vh)
print("Writing results in file ... ")

for i, row in test.iterrows():
    try:
        x = users.index(row['userID'])
        try:
            y = movies.index(row['movieID'])
            pred = new_matrix[x,y]
            f.write(str(round(pred,1)))
            f.write("\n")
        except ValueError:
            #Movie not recognized
            pred = np.mean(new_matrix[x,:])
            f.write(str(round(pred,1)))
            f.write("\n")
    except ValueError:
        #User not recornize
        try:
            y = movies.index(row['movieID'])
            pred = np.mean(new_matrix[:,y])
            f.write(str(round(pred,1)))
            f.write("\n")
        except ValueError:
            #Movie and user not recognized, random value
            pred = np.random.uniform(3,4)
            f.write(str(round(pred,1)))
            f.write("\n")
            
print("Number of predictions: ", i)          
f.close()