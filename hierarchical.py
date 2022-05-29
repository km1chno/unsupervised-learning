import numpy as np 
from matplotlib import pyplot as plt
from sortedcontainers import SortedList

def dist(x, z):
    return np.linalg.norm(x-z)

class HierarchicalClustering: 
    def __init__(self, X):
        self.X = np.copy(X)

    def scale_data(self):
        v = np.asarray(np.var(self.X, axis = 0)).flatten()
        m = np.asarray(np.mean(self.X, axis = 0)).flatten()
        for j in range(0, self.n):
            self.X[:, j] = (self.X[:, j] - m[j]) / np.sqrt(v[j])

    def get_coeffs(self, x, y, method):
        a = self.cluster_size[x]
        b = self.cluster_size[y]
        c = a+b
        if method == 0:     #single
            return 1/2, 1/2, 0, -1/2
        elif method == 1:   #complete
            return 1/2, 1/2, 0, 1/2
        elif method == 2:   #average
            return a/(a+b), b/(a+b), 0, 0
        elif method == 3:   #centroid
            return a/(a+b), b/(a+b), (-a*b)/((a+b)**2), 0
        elif method == 4:   #ward 
            return (a+c)/(a+b+c), (b+c)/(a+b+c), (-c)/(a+b+c), 0
            

    def doit(self, method, nclusters):
        self.nclusters = nclusters
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        D = np.zeros((2*self.m, 2*self.m), dtype=float)
        self.cluster_size = np.ones(2*self.m)
        self.cluster_id = np.zeros(self.m, dtype=int)
        deleted = np.zeros(2*self.m)
        dist_sorted = SortedList()

        self.scale_data()
        
        print(f'initialization...')
        for i in range(self.m):
            self.cluster_id[i] = i
            for j in range(i+1, self.m):
                print(f'init {i} {j}')
                d = dist(self.X[i], self.X[j])
                if method == 4:
                    d = d**2
                D[i][j] = D[j][i] = d
                dist_sorted.add((D[i][j], i, j))

        for k in range(self.m, 2*self.m - nclusters):
            print(f'combining clusters {k-self.m}/{self.m-nclusters-1}')

            smallest_dist, x, y = dist_sorted[0]
            dist_sorted.discard((smallest_dist, x, y))
            self.cluster_size[k] = self.cluster_size[x] + self.cluster_size[y]
            ax, ay, b, g = self.get_coeffs(x, y, method)
            for i in range(k):
                if i != x and i != y and deleted[i] == 0:
                    dist_sorted.discard((D[i][x], i, x))
                    dist_sorted.discard((D[i][x], x, i))
                    dist_sorted.discard((D[i][y], i, y))
                    dist_sorted.discard((D[i][y], y, i))
                    D[i][k] = D[k][i] = ax*D[i][x] + ay*D[i][y] + b*D[x][y] + g*np.abs(D[i][x] - D[i][y])
                    dist_sorted.add((D[i][k], i, k))
            deleted[x] = deleted[y] = 1
            self.cluster_id[(self.cluster_id == y)|(self.cluster_id == x)] = k

        scale_array = np.sort(np.unique(self.cluster_id))
        for i in range(self.m):
            self.cluster_id[i] = np.where(scale_array == self.cluster_id[i])[0]

        return self.X, self.cluster_id