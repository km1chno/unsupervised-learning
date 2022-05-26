import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from hierarchical import HierarchicalClustering
from kmeans import KMeansClustering
import sys 

colors = [
    "#dc143c", #crimson
    "#7cfc00", #lawngreen
    "#1e90ff", #dodgerblue
    "#ff4500", #orangered
    "#8b008b", #darkmagenta
    "#ffdead", #navajowhite
    "#696969", #dimgray
    "#8b4513", #saddlebrown
    "#228b22", #forestgreen
    "#808000", #olive
    "#483d8b", #darkslateblue
    "#008b8b", #darkcyan
    "#9acd32", #yellowgreen
    "#00008b", #darkblue
    "#8fbc8f", #darkseagreen
    "#b03060", #maroon3
    "#ff8c00", #darkorange
    "#ffff00", #yellow
    "#8a2be2", #blueviolet
    "#00ff7f", #springgreen
    "#00ffff", #aqua
    "#00bfff", #deepskyblue
    "#0000ff", #blue
    "#b0c4de", #lightsteelblue
    "#ff00ff", #fuchsia
    "#fa8072", #salmon
    "#90ee90", #lightgreen
    "#ff1493", #deeppink
    "#7b68ee", #mediumslateblue
    "#ee82ee", #violet
    "#ffb6c1" #lightpink
]

def test_hierarchical(X, y):
    hcluster = HierarchicalClustering(X, y)
    hcluster.doit(0, 0, 2)

def test_kmeans(X, y, min_clusters, max_clusters):
    kmeans = KMeansClustering(X, y)
    m = X.shape[0]
    n = X.shape[1]
    scaled_X, k, clustering, C, smallest_error, sil = kmeans.doit(min_clusters, max_clusters, use_sil=True)

    fig = plt.figure(figsize = (25, 12), dpi= 80)
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(range(min_clusters, max_clusters+1), smallest_error, color="#8a2be2")
    for i in range(min_clusters, max_clusters+1):
        ax.scatter(i, smallest_error[i-min_clusters], s=80, color="#8a2be2")
    ax.set_title("k-means++ error")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    bx = fig.add_subplot(2, 2, 2)
    bx.plot(range(min_clusters, max_clusters+1), sil, color="#8a2be2")
    for i in range(min_clusters, max_clusters+1):
        bx.scatter(i, sil[i-min_clusters], s=80, color="#8a2be2")
    bx.set_title("k-means++ silhouette coeffs")
    bx.xaxis.set_major_locator(MaxNLocator(integer=True))

    cx = fig.add_subplot(2, 2, 3)
    for j in range(m):
        cx.scatter(scaled_X[j][0], scaled_X[j][1], color=colors[clustering[j]])
    for j in range(k):
        cx.scatter(C[j][0], C[j][1], color=colors[j], marker="P", edgecolors='black', s=400)
    cx.set_title("k-means++ for " + str(k) + " clusters")

    dx = fig.add_subplot(2, 2, 4)
    for j in range(m):
        dx.scatter(scaled_X[j][0], scaled_X[j][1], color=colors[int(y[j])-1])
    real_clusters = int(np.amax(y) - np.amin(y)) + 1
    dx.set_title("real classification for " + str(real_clusters) + " clusters")

    fig.tight_layout()
    plt.show()


i = sys.argv[1]
data = np.loadtxt(f"data/dane_2D_{i}.txt", dtype = float)
X, y = np.split(data, [2], axis = 1)

plt.rcParams['axes.grid'] = True
test_kmeans(X, y, 2, 8)
