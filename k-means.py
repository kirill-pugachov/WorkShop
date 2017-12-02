# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:48:14 2017

@author: Kirill
"""

import matplotlib.pyplot as plt
import numpy as np


k = 5
points = np.vstack(((np.random.randn(150, 2) * 0.65 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.35 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def main(points):
    num_iterations = 100
        
    # Initialize centroids
    centroids = initialize_centroids(points, k)
    
    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
    
    return centroids


if __name__ == "__main__":
    
    
    centroids = main(points)
    
    centroids = initialize_centroids(points, k)

    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    ax = plt.gca()
    
    centroids = points.copy()
    np.random.shuffle(centroids)
    centroids[:3]
    temp = closest_centroid(points, centroids)
    temp[1]