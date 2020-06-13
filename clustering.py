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
        metric : algorithm to calculate distance 'euclidean' or 'manhattan'
        verbose : log the clustering process. 0 to turn on log, 1 to turn off log
        max_iter : maximum iteration of clustering process
    """

    def __init__(self, x, centroid, ncluster, metric='euclidean', verbose=0, max_iter=1000):
        self.x = x
        self.centroid = centroid
        self.ncluster = ncluster
        self.metric = metric
        self.verbose = verbose
        self.max_iter = max_iter

    def distance(self, new_centroid):
        # return euclidean distance between data and centroid
        if (self.metric=='euclidean'):
            dist = np.linalg.norm(new_centroid[:, np.newaxis] - self.x, axis=2)
        elif (self.metric=='manhattan'):
            dist = (np.abs(self.x[:,0,None] - new_centroid[:,0]) + np.abs(self.x[:,1,None] - new_centroid[:,1])).T
        return dist 
    
    def clustering(self):
        # create first cluster. we don't care about the size and value, since it's only for checking first iteration
        cluster = np.array([-1]).astype(np.int8)
        # create centroid from user input and change data type
        new_centroid = self.centroid.astype(np.float32)
        n_iter = 1
        while True:
            # calculate distance between data and centroid
            dist = KMeans.distance(self, new_centroid)
            if (self.verbose==1):
                print('Iteration : {}'.format(n_iter))
                print('   Distance :' )
                print(dist.T)
            # find cluster based on minimum distance 
            new_cluster = np.argmin(dist, axis=0).astype(np.int8)
            if (self.verbose==1):
                print('   Cluster :' )
                print(new_cluster)
            # looping for find new centroid for each cluster
            for i in range (self.ncluster):
                # find centroid for i-th cluster
                new_centroid[i,:] = np.expand_dims(np.mean(self.x[np.argwhere(new_cluster==i).reshape(-1),:], axis=0), axis=0)
            if (self.verbose==1):
                print('   Centroid :' )
                print(new_centroid,'\n')
            # break the loop if new cluster same as previous cluster
            if (np.array_equal(new_cluster,cluster)==True) or (n_iter>self.max_iter):
                break
            else:
                cluster = new_cluster
                n_iter += 1

        return new_cluster, new_centroid


class KMedoids():
    """ Implementation of KMedoids algorithm
    '''Params
        x : 2d numpy array. row is number of data, column is number of features
        medoid : 2d numpy array. row is number of medoid, column is number of features
        ncluster : number of cluster
        metric : algorithm to calculate distance 'euclidean' or 'manhattan'
        verbose : log the clustering process. 0 to turn on log, 1 to turn off log
        max_iter : maximum iteration of clustering process
    """

    def __init__(self, x, ncluster,  metric='euclidean', verbose=0, max_iter=1000):
        self.x = x
        self.ncluster = ncluster
        self.metric = metric
        self.medoid = self.x[np.random.choice(self.x.shape[0], self.ncluster, replace=False)]        
        self.verbose = verbose
        self.max_iter = max_iter

    def distance(self, new_medoid):
        # return euclidean distance between data and centroid
        if (self.metric=='euclidean'):
            dist = np.linalg.norm(new_medoid[:, np.newaxis] - self.x, axis=2)
        elif (self.metric=='manhattan'):
            dist = (np.abs(self.x[:,0,None] - new_medoid[:,0]) + np.abs(self.x[:,1,None] - new_medoid[:,1])).T
        return dist 
    
    def clustering(self):
        # create first cluster. we don't care about the size and value, since it's only for checking first iteration
        cluster = np.array([-1]).astype(np.int8)
        # create centroid from user input and change data type
        new_medoid = self.medoid.astype(np.float32)
        n_iter = 1
        while True:
            # calculate distance between data and centroid
            dist = KMeans.distance(self, new_medoid)
            if (self.verbose==1):
                print('Iteration : {}'.format(n_iter))
                print('   Distance :' )
                print(dist.T)
            # find cluster based on minimum distance 
            new_cluster = np.argmin(dist, axis=0).astype(np.int8)
            if (self.verbose==1):
                print('   Cluster :' )
                print(new_cluster)
            # looping for find new centroid for each cluster
            for i in range (self.ncluster):
                # find centroid for i-th cluster
                new_medoid[i,:] = np.expand_dims(np.mean(self.x[np.argwhere(new_cluster==i).reshape(-1),:], axis=0), axis=0)
            if (self.verbose==1):
                print('   Medoid :' )
                print(new_medoid,'\n')
            # break the loop if new cluster same as previous cluster
            if (np.array_equal(new_cluster,cluster)==True) or (n_iter>self.max_iter):
                break
            else:
                cluster = new_cluster
                n_iter += 1

        return new_cluster, new_medoid

"""
data test

dt = np.array([[1,2],[2,3],[4,3],[5,4]])
ncl = 2
centroid = np.array([[1,0],[4,4]])

l,k = KMeans(dt, centroid, ncl, metric='manhattan', verbose=1).clustering()
l,k = KMedoids(dt, ncl, verbose=1).clustering()
"""