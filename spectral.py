import numpy as np 
from matplotlib import pyplot as plt
from sortedcontainers import SortedList
from kmeans import KMeansClustering
from scipy.sparse import csgraph

INF = 10**10

def dist_inv(x):
    if x == 0:
        return INF
    return 1/x

def get_gauss(t):
    return lambda x : np.exp(-(x**2)/(t**2))

def dist(x, z):
    return np.linalg.norm(x-z)

class SpectralClustering:
    def __init__(self, X):
        self.X = np.copy(X)

    def scale_data(self):
        v = np.asarray(np.var(self.X, axis = 0)).flatten()
        m = np.asarray(np.mean(self.X, axis = 0)).flatten()
        for j in range(0, self.n):
            self.X[:, j] = (self.X[:, j] - m[j]) / np.sqrt(v[j])

    def doit(self, graph_method, min_clusters=2, max_clusters=5, weight_function=dist_inv, max_dist=0, neighbors=0):
        self.n = self.X.shape[1]
        self.m = self.X.shape[0]
        self.scale_data()

        A = np.zeros((self.m, self.m), dtype=float)
        D = np.zeros((self.m, self.m), dtype=float)
        if graph_method == 2:
            for i in range(self.m):
                dist_array = np.zeros(self.m)
                for j in range(self.m):
                    dist_array[j] = dist(self.X[i], self.X[j])
                dist_array[i] = INF
                for k in range(neighbors):
                    j = np.argmin(dist_array)
                    A[i][j] = A[j][i] = weight_function(dist_array[j])
                    dist_array[j] = INF
                for j in range(self.m):
                    D[i][i] += A[i][j]
        else:
            for i in range(self.m):
                for j in range(self.m):
                    d = dist(self.X[i], self.X[j])
                    if i == j:
                        A[i][j] = 0
                    elif graph_method == 0:
                        A[i][j] = weight_function(dist(self.X[i], self.X[j]))
                    elif graph_method == 1 and d < max_dist:
                        A[i][j] = 1
                    D[i][i] += A[i][j]
        
        print(f'Calculating normed laplacian of similarity graph...')
        L = D - A
        for i in range(self.m):
            if D[i][i] > 0:
                D[i][i] = 1/np.sqrt(D[i][i])
        L_norm = np.dot(np.dot(D, L), D)

        print(f'Calculating eigenvalues and eigenvectors of laplacian...')
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        n_clusters = max_clusters-min_clusters+1
        errors = np.zeros(n_clusters)
        sil = np.zeros(n_clusters)
        best_k = min_clusters
        best_clustering = np.zeros(self.m)
        for k in range(min_clusters, max_clusters+1):
            new_X = eigenvectors[:,:k]
            kmeans = KMeansClustering(new_X)
            scaled_X, kk, clustering, C, smallest_error, single_sil = kmeans.doit(k, k, use_sil=True)

            errors[k-min_clusters] = smallest_error[0]
            sil[k-min_clusters] = single_sil[0]
            if sil[k-min_clusters] >= sil[best_k-min_clusters]:
                best_k = k 
                best_clustering = np.copy(clustering)

        return self.X, best_k, best_clustering, errors, sil, A



