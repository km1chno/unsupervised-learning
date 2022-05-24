import numpy as np 
from matplotlib import pyplot as plt
from sortedcontainers import SortedList

INF = 10**10

metrics = [
    lambda x, z: np.linalg.norm(x-z)
]

colors = ["#00bfa0", "#0c06bf", "#bf0606", "#12bf06", "#bf06bf", "#b9bf06", "#06b3bf", "#000000", "#b3d4ff"]

class HierarchicalClustering: 
    def __init__(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)

    def get_coeffs(self, x, y, method):
        return 1/2, 1/2, 0, -1/2

    def doit(self, metric, method, nclusters):
        metric_func = metrics[metric]
        self.m = self.X.shape[0]
        self.D = np.zeros((2*self.m, 2*self.m), dtype=float)
        cluster_id = np.zeros(self.m, dtype=int)
        self.deleted = np.zeros(2*self.m)

        dist_sorted = SortedList()
        for i in range(self.m):
            cluster_id[i] = i
            for j in range(i+1, self.m):
                print(f'init pair ({i}, {j})')

                self.D[i][j] = self.D[j][i] = metric_func(self.X[i], self.X[j])
                dist_sorted.add((self.D[i][j], i, j))

        for k in range(self.m, 2*self.m - nclusters):
            print(f'combining clusters {k-self.m}/{self.m-nclusters-1}')

            smallest_dist, x, y = dist_sorted[0]
            dist_sorted.discard((smallest_dist, x, y))
            ax, ay, b, g = self.get_coeffs(x, y, method)
            for i in range(k):
                if i != x and i != y and self.deleted[i] == 0:
                    dist_sorted.discard((self.D[i][x], i, x))
                    dist_sorted.discard((self.D[i][x], x, i))
                    dist_sorted.discard((self.D[i][y], i, y))
                    dist_sorted.discard((self.D[i][y], y, i))
                    self.D[i][k] = self.D[k][i] = ax*self.D[i][x] + ay*self.D[i][y] + b*self.D[x][y] + g*np.abs(self.D[i][x] - self.D[i][y])
                    dist_sorted.add((self.D[i][k], i, k))
            self.deleted[x] = self.deleted[y] = 1
            cluster_id[(cluster_id == y)|(cluster_id == x)] = k

        fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi= 80)
        plt.gca().spines["top"].set_alpha(0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(0)
        plt.gca().spines["left"].set_alpha(.3)
        plt.grid()

        next_color = 1
        cluster_color = np.zeros(2*self.m-nclusters, dtype=int)
        for i in range(self.m):
            if cluster_color[cluster_id[i]] == 0:
                cluster_color[cluster_id[i]] = next_color
                next_color += 1
            ax.scatter(self.X[i][0], self.X[i][1], color=colors[cluster_color[cluster_id[i]]])

        plt.show()
        