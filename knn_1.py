# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:25:22 2017

@author: Kirill
"""

import pandas as pd
import numpy as np
from math import sqrt
import operator


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
proportion = 0.8
k = 7

def get_data(data_url):
    data_set = pd.read_csv(data_url, header=None)
    
    return data_set


def split_data(data_set, proportion):
    '''
    data preparing
    '''
    data_set = data_set.iloc[np.random.permutation(len(data_set))]
    data_train = data_set[:int(proportion * len(data_set))]
    data_test = data_set[int(proportion * len(data_set)):]
    
    return data_train, data_test


def sqrt_dist(data_1, data_2):
    '''
    count distance
    '''
    dist = [(i-j)**2 for i,j in list(zip(data_1, data_2))]
    
    return sqrt(sum(dist))
    

def get_neighbours(instance, Data_train, k):
    '''
    get nearest points by distance
    '''
    distances = []
    for i in Data_train.iloc[:,:-1].values:
        distances.append(sqrt_dist(instance, i))
    distances = tuple(zip(distances, Data_train[Data_train.columns[:-1]].values))
    
    return sorted(distances, key=operator.itemgetter(0))[:k]


def get_labels(neigbours):
    '''
    return for point class label
    '''
    label = []
    for item in neigbours:
        for ind in Data_train[Data_train.columns[:-1]][Data_train[Data_train.columns[:-1]] == item[1]].dropna().index:
            label.append(Data_train.iloc(ind)[-1][len(Data_train.iloc(ind)[-1]) - 1])      
    return label 


def get_response(neigbours):
    labels = get_labels(neigbours)
    lab = get_labels(neigbours)
    if len(set(labels)) == 1:
        return labels[0]
    elif len(set(labels)) > 1:
        result = {}
        for mark in lab:
            if mark in result:
                result[mark] += 1
            else:
                result[mark] = 1
        return max(result, key=result.get)


def get_predictions(Data_train, Data_test, k):
    predictions = []
    for i in Data_test.iloc[:,:-1].values:
        neigbours = get_neighbours(i, Data_train, k)
        response = get_response(neigbours)
        predictions.append(response)
        
    return predictions


def mean(instance):
    
    return sum(instance)/len(instance)


def get_accuracy(Data_test,predictions):
    '''
    count accuracy
    '''
    return mean([i == j for i,j in zip(Data_test[Data_test.columns[-1]].values, predictions)])


if __name__ == "__main__":
    df = get_data(data_url)
    Data_train, Data_test = split_data(df, proportion) 
    accuracy = get_accuracy(Data_test, get_predictions(Data_train, Data_test, 5))