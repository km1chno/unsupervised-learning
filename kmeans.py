import numpy as np 
from matplotlib import pyplot as plt
from sortedcontainers import SortedList

def dist(x, z):
    return np.linalg.norm(x-z)

RUNS = 10
MIN_CENTROID_CHANGE = 10**(-3)
INF = 10**10

class KMeansClustering: 
    def __init__(self, X):
        self.X = np.copy(X)

    def scale_data(self):
        v = np.asarray(np.var(self.X, axis = 0)).flatten()
        m = np.asarray(np.mean(self.X, axis = 0)).flatten()
        for j in range(0, self.n):
            self.X[:, j] = (self.X[:, j] - m[j]) / np.sqrt(v[j])

    def find_closest_centroid(self, i, C):
        best_dist = INF
        best_cent = -1
        for cent in range(self.ncentroids):
            d = dist(C[cent], self.X[i])
            if d < best_dist:
                best_dist = d 
                best_cent = cent
        return best_cent, best_dist

    def init_centroids(self, k):
        first = np.random.randint(0, self.m)
        C = np.zeros((k, self.n))
        C[0] = np.copy(self.X[first])
        self.ncentroids = 1
        p = np.zeros(self.m)
        for j in range(1, self.k):
            for i in range(self.m):
                cent, best_dist = self.find_closest_centroid(i, C)
                p[i] = best_dist ** 2
            p /= np.sum(p)
            next_cent = np.random.choice(range(self.m), p=p)
            C[j] = np.copy(self.X[next_cent])
            self.ncentroids += 1
        return C

    def calc_error(self, cluster_id, C): 
        error = 0 
        for i in range(self.m):
            error += dist(self.X[i], C[cluster_id[i]]) ** 2
        return error

    def calc_sil(self, k, D, min_clusters, best_clustering):
        print(f'calculating silhouette coefficient for k = {k}...')

        cluster_size = np.zeros(k)
        for i in range(self.m):
            cluster_size[best_clustering[k-min_clusters][i]] += 1
        S = 0
        for i in range(self.m):
            dist_from_cluster = np.zeros(k)
            for j in range(self.m):
                if i != j:
                    dist_from_cluster[best_clustering[k-min_clusters][j]] += D[i][j]
            a = 0
            if cluster_size[best_clustering[k-min_clusters][i]] > 1:
                a = dist_from_cluster[best_clustering[k-min_clusters][i]] / (cluster_size[best_clustering[k-min_clusters][i]]-1)
            b = INF
            for j in range(k):
                if j != best_clustering[k-min_clusters][i]:
                    b = np.amin([b, dist_from_cluster[j]/cluster_size[j]])
            S += (b-a)/np.amax([a, b])
        return S/self.m

    def doit(self, min_clusters, max_clusters, use_sil=False, scale=True):
        self.n = self.X.shape[1]
        self.m = self.X.shape[0]
        if scale:
            self.scale_data()
        n_clusters = max_clusters-min_clusters+1
        cluster_id = np.zeros(self.m, dtype=int) 
        smallest_error = np.full(n_clusters, INF)
        best_clustering = np.zeros((n_clusters, self.m), dtype=int)
        best_C = np.zeros((n_clusters, max_clusters, self.n))

        for k in range(min_clusters, max_clusters+1):
            print(f'k-means for {k} clusters...')
    
            self.k = k
            for run in range(RUNS):
                print(f'run {run}/{RUNS}')
                C = self.init_centroids(k)
                while True:
                    new_C = np.zeros((k, self.n))
                    cluster_size = np.zeros(k)
                    for i in range(self.m):
                        j, best_dist = self.find_closest_centroid(i, C)
                        cluster_size[j] += 1
                        new_C[j] += self.X[i]
                        cluster_id[i] = j

                    change = 0
                    for j in range(k):
                        new_C[j] /= cluster_size[j]
                        if np.abs(np.linalg.norm(new_C[j] - C[j])) >= MIN_CENTROID_CHANGE:
                            change = 1
                        C[j] = np.copy(new_C[j])
                    if change == 0:
                        break
                
                error = self.calc_error(cluster_id, C)
                if error < smallest_error[k-min_clusters]:
                    smallest_error[k-min_clusters] = error 
                    best_clustering[k-min_clusters] = np.copy(cluster_id)
                    for j in range(k):
                        best_C[k-min_clusters][j] = np.copy(C[j])

        best_k = min_clusters
        sil = np.zeros(n_clusters)
        if use_sil:
            D = np.zeros((self.m, self.m))
            for i in range(self.m):
                for j in range(self.m):
                    D[i][j] = dist(self.X[i], self.X[j])
            
            for k in range(min_clusters, max_clusters+1):
                sil[k-min_clusters] = self.calc_sil(k, D, min_clusters, best_clustering)
                if sil[k-min_clusters] >= sil[best_k-min_clusters]:
                    best_k = k
        else:
            whole_change = smallest_error[0] - smallest_error[n_clusters-1]    
            for j in range(min_clusters, max_clusters+1):
                if smallest_error[j-min_clusters] - smallest_error[j+1-min_clusters] <= whole_change / n_clusters:
                    best_k = j
                    break

        return self.X, best_k, best_clustering[best_k-min_clusters], best_C[best_k-min_clusters], smallest_error, sil
        