# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:23:00 2018

@author: Ushma
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
Insample_score=[]
Out_of_sample_score=[]

start = time.clock()

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=5)
    tree.fit(X_train, y_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = tree.predict(X_train)    
    Ins_score = metrics.accuracy_score(y_train, y_train_pred)
    Op_score = metrics.accuracy_score(y_test, y_test_pred)
    Insample_score.append(Ins_score)    
    Out_of_sample_score.append(Op_score)
    print('Random State: %d, In-sample score: %.3f, Out of sample score: %.3f' % (i,Ins_score,Op_score))

print("Insample mean:",np.mean(Insample_score),"std deviation:",np.std(Insample_score))
print("Out_of_sample mean:",np.mean(Out_of_sample_score),"std deviation:",np.std(Out_of_sample_score))

end = time.clock()

print ("Time Required: %.2f sec" % (end-start))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

start = time.clock()

for i in range(1,11): 
    kfold = KFold(n_splits=10,random_state=1).split(X_train, y_train)
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=i)
    print("Random State %d" %i)
    scores = cross_val_score(estimator=tree,X=X_train,y=y_train,cv=10,n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy mean/std: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
end = time.clock()

print ("Time Taken: %.2f sec" % (end-start))


start = time.clock()
for i in range(1,11):
    kfold=KFold(n_splits=5,random_state=i).split(X_train,y_train)
    tree.fit(X_train,y_train)
    cv_score=cross_val_score(estimator=tree,X=X_train,y=y_train,cv=10,n_jobs=1)
    y_pred_test=tree.predict(X_test)
    out_score=metrics.accuracy_score(y_test,y_pred_test)
    print("Random State %d" %i)
    print("CV accuracy scores:", cv_score)
    print("CV mean:",np.mean(cv_score),", std deviation:", np.std(cv_score),)
    print("Out of sample score:", out_score,'\n')

end = time.clock()
print ("Time Required: %.2f sec" % (end-start))


print("My name is Ushma Bhatt")
print("My NetID is: ushmab2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")