import numpy as np
import operator
from random import randrange
from math import log
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
labels=[1,2,3,4]
# X_train_data=np.load('train_data.npy')
# y_train_data=np.load('train_target.npy')
# np.random.seed(0)
# order = np.random.permutation(len(X_train_data))
# X_train=X_train_data[order]
# y_train=y_train_data[order]

def get_subDataSet(dataSet,ratio):
    subDataSet=[]
    lenSubData=round(len(dataSet)*ratio)
    while len(subDataSet)<lenSubData:
        index=randrange(len(dataSet)-1)
        subDataSet.append(dataSet[index])
    print(len(subDataSet))
    return subDataSet

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseFeat(dataSet,n_feature):
    baseEnt=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeat=-1

    features=[]
    while len(features)<n_feature:
        index=randrange(len(dataSet[0])-1)
        if index not in features:
            features.append(index)

    for index in features:
        featList=[example[index] for example in dataSet]
        uniqueVs=set(featList)
        newEnt=0.0
        for value in uniqueVs:
            subDataSet=splitDataSet(dataSet,index,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEnt+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEnt-newEnt
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeat=index
    return bestFeat

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.itertiems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(X,y,labels,n_feature=3):
    classList=y
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(X[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseFeat(X,n_feature)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in X]
    uniqueVs=set(featValues)
    for value in uniqueVs:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(X,bestFeat,value),subLabels)
    return myTree

#test again