"""
created by Made Satria Wibawa 2020
"""

import numpy as np 

class KMeans():
    """ Implementation of KMeans algorithm
    '''Params
        x : 2d numpy array. row is number of data, column is number of features
        centroid : 2d numpy array. row is number of centroid, column is number of features
        ncluster : number of cluster
    """

    def __init__(self, x, centroid, ncluster):
        self.x = x
        self.centroid = centroid
        self.ncluster = ncluster

    def distance(self):
        # return euclidean distance between data and centroid
        return np.linalg.norm(self.centroid[:, np.newaxis] - self.x, axis=2)
    
    def clustering(self):
        # create first cluster. we don't care about the size and value, since it's only for checking first iteration
        cluster = np.array([-1]).astype(np.int8)
        # create centroid from user input and change data type
        new_centroid = self.centroid.astype(np.float32)
        while True:
            # calculate distance between data and centroid
            dist = KMeans.distance(self)
            # find cluster based on minimum distance 
            new_cluster = np.argmin(dist, axis=0).astype(np.int8)
            # looping for find new centroid for each cluster
            for i in range (self.ncluster):
                # find centroid for i-th cluster
                new_centroid[i,:] = np.expand_dims(np.mean(self.x[np.argwhere(new_cluster==i).reshape(-1),:], axis=0), axis=0)
            # break the loop if new cluster same as previous cluster
            if (np.array_equal(new_cluster,cluster)==True):
                break
            else:
                cluster = new_cluster

        return new_cluster, new_centroid

class KMedoids():
    """ Implementation of KMedoids algorithm
    '''Params
        x : 2d numpy array. row is number of data, column is number of features
        ncluster : number of cluster
    """

    def __init__(self, x, ncluster):
        self.x = x
        self.centroid = self.x[np.random.choice(self.x.shape[0], ncluster, replace=False)]
        self.ncluster = ncluster

    def distance(self):
        # return euclidean distance between data and centroid
        return np.linalg.norm(self.centroid[:, np.newaxis] - self.x, axis=2)
    
    def clustering(self):
        # create first cluster. we don't care about the size and value, since it's only for checking first iteration
        cluster = np.array([-1]).astype(np.int8)
        # create centroid from user input and change data type
        new_centroid = self.centroid.astype(np.float32)
        while True:
            # calculate distance between data and centroid
            dist = KMedoids.distance(self)
            # find cluster based on minimum distance 
            new_cluster = np.argmin(dist, axis=0).astype(np.int8)
            # looping for find new centroid for each cluster
            for i in range (self.ncluster):
                # find centroid for i-th cluster
                new_centroid[i,:] = np.expand_dims(np.mean(self.x[np.argwhere(new_cluster==i).reshape(-1),:], axis=0), axis=0)
            # break the loop if new cluster same as previous cluster
            if (np.array_equal(new_cluster,cluster)==True):
                break
            else:
                cluster = new_cluster

        return new_cluster, new_centroid