# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:47:18 2019

@author: Laura
"""

# Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler 
#from random import sample 
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


   
# DATA
# There are 14,528 instances and 54 features in the training set
# Specifically, attributes #1, 8, 9, 20, 22, 31, 42, 47, 50, 54 are numeric, 
# and the rest are all binary (except the one for class labels).
# #55 is the label
path_train = "train.csv"
path_test = "test.csv"

#We open our file where we are going to write our results
f = open("FinalResults.dat", "w+")

# Rango is a set od numbers that will represent the feauters of the data
# There is no info available about the nature of the data so the features are represent as a numericla value
rango_train = list(range(1, 56))
rango_test = list(range(1, 55))

# This variables are not binary but numerical
num_variables = [1, 8, 9, 20, 22, 31, 42, 47, 50, 54]

# First the data is uploaded
df = pd.read_csv(path_train, names = rango_train)
df_t = pd.read_csv(path_test, names = rango_test)


# Last column that contains the label is stored in an auxiliary structure
y_train = df[55]
# Labels are delete from the dataset
df = df.drop(labels=55, axis=1)
X_train = df
X_test = df_t

from sklearn.neighbors import KNeighborsClassifier
mlp = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', p=2)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

for i in predictions:
    x = str(i)
    f.write(x)
    f.write("\n")
f.close()

#CODE USE



#print("1KNN",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))

#X_test = StandardScaler().fit_transform(df_t)
#dfst = pd.DataFrame(X_test, index=df_t.index, columns=df_t.columns)
#X_test = df_t
#minn = 0
#min_t = 0


#Transform data     
# Fitting the PCA algorithm with our Data
#pca = PCA().fit(dfs)
## Plotting the Cumulative Summation of the Explained Variance
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Pulsar Dataset Explained Variance')
#plt.show()

# The conclusion is that we still need at least 45 features to ger a .99 of variance
#pca = PCA(n_components=45)
#X_train = pca.fit_transform(dfs)
#X_test = pca.fit_transform(dfst)

#Splitting the data
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

#PREPROCESSING FOR OTHER CLASSIFIERS
#X_train_2 = StandardScaler().fit_transform(X_train)
#
#X_test_2 = StandardScaler().fit_transform(X_test)
#
#X_train_3 = X_train
#X_test_3 = X_test
#
#for i, row in X_train_3.iterrows():
#    if (row[1] < minn):
#        minn = row[1]
#
#for i, row in X_test_3.iterrows():
#    if (row[1] < min_t):
#        min_t = row[1]
#        
#minn = abs(minn)
#
#min_t = abs(min_t)
#for i, row in X_train_3.iterrows():
#    row[1] = row[1] + minn
#
#for i, row in X_test_3.iterrows():
#    row[1] = row[1] + min_t
##    
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()
#clf = clf.fit(X_train_3, y_train)
#predictions = clf.predict(X_test_3)
#print("MULT NB",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#from sklearn.naive_bayes import ComplementNB
#clf = ComplementNB()
#clf = clf.fit(X_train_3, y_train)
#predictions = clf.predict(X_test_3)
#print("COMPLEMENT NB",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf = clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)
#print("GAUSSIAN NBNOESCALADO",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#
#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(activation='logistic',max_iter= 500)
#mlp.fit(X_train_2,y_train)
#predictions = mlp.predict(X_test_2)
#print("1NN",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#
#from sklearn.svm import SVC
#clf = SVC(class_weight={1:1.7,2:1,3:4.5,4:7,5:6,6:5,7:5.5})
#clf = clf.fit(X_train_2, y_train)
#predictions = clf.predict(X_test_2)
#print("1SVC",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#
#from sklearn import tree
#clf = tree.DecisionTreeClassifier(class_weight={1:1.7,2:1,3:4.5,4:7,5:6,6:5,7:5.5})
#clf = clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)
#print("1Tree",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(class_weight={1:1.7,2:1,3:4.5,4:7,5:6,6:5,7:5.5})
#clf = clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)
#print("1Tree",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(class_weight={1:1.7,2:1,3:4.5,4:7,5:6,6:5,7:5.5})
#clf = clf.fit(X_train_2, y_train)
#predictions = clf.predict(X_test_2)
#print("1Tree",confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
