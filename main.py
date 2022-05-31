import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from hierarchical import HierarchicalClustering
from kmeans import KMeansClustering
from spectral import SpectralClustering
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

"""
colors = [
    "#7f0000",
    "#006400",
    "#808000",
    "#483d8b",
    "#008080",
    "#4682b4",
    "#9acd32",
    "#00008b",
    "#32cd32",
    "#daa520",
    "#8fbc8f",
    "#8b008b",
    "#b03060",
    "#d2b48c",
    "#ff4500",
    "#00ced1",
    "#ff8c00",
    "#ffff00",
    "#00ff00",
    "#00fa9a",
    "#8a2be2",
    "#dc143c",
    "#f4a460",
    "#0000ff",
    "#adff2f",
    "#da70d6",
    "#ff00ff",
    "#1e90ff",
    "#f0e68c",
    "#fa8072",
    "#dda0dd",
    "#87ceeb",
    "#ff1493",
    "#7b68ee",
    "#7fffd4",
    "#ffb6c1"
]
"""

def calc_classification_error(X, y, clustering):
    m = X.shape[0]
    good = 0
    for i in range(m):
        for j in range(i+1, m):
            if clustering[i] == clustering[j] and y[i] == y[j]:
                good += 1
            if clustering[i] != clustering[j] and y[i] != y[j]:
                good += 1
    return 1 - good/(m*(m-1)/2)

method_subtitle = [
    "(Single linkage)",
    "(Complete linkage)",
    "(Group average)",
    "(Centroid)",
    "(Ward)"
]

def test_hierarchical(X, y, method, nclusters):
    hcluster = HierarchicalClustering(X)
    scaled_X, clustering = hcluster.doit(method, nclusters)
    m = X.shape[0]

    return scaled_X, clustering, calc_classification_error(X, y, clustering)

    fig = plt.figure(figsize = (12, 6), dpi= 80)
    cx = fig.add_subplot(1, 2, 1)
    for j in range(m):
        cx.scatter(scaled_X[j][0], scaled_X[j][1], color=colors[clustering[j]])

    cx.set_title("Hierarchical Clustering for " + str(nclusters) + " clusters " + method_subtitle[method])

    dx = fig.add_subplot(1, 2, 2)
    for j in range(m):
        dx.scatter(scaled_X[j][0], scaled_X[j][1], color=colors[int(y[j])-1])
    real_clusters = int(np.amax(y) - np.amin(y)) + 1
    dx.set_title("Actual classification for " + str(real_clusters) + " clusters")

    print(f'classification error: {calc_classification_error(X, y, clustering)}')

    fig.tight_layout()
    plt.show()


def test_kmeans(X, y, min_clusters, max_clusters, file, image):
    kmeans = KMeansClustering(X)
    m = X.shape[0]
    n = X.shape[1]
    scaled_X, k, clustering, C, smallest_error, sil = kmeans.doit(min_clusters, max_clusters, use_sil=True)

    fig = plt.figure(figsize = (12,12), dpi= 80)
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
    dx.set_title("Actual classification for " + str(real_clusters) + " clusters")
    
    fig.tight_layout()

    plt.savefig(image)
    file.write(f'classification error for kmeans: {calc_classification_error(X, y, clustering)}\n')
    #plt.show()


def test_spectral(X, y, graph_method, min_clusters=2, max_clusters=8, max_dist=0, neighbors=0, weight_function=0):
    spectral = SpectralClustering(X)
    m = X.shape[0]
    n = X.shape[1]
    scaled_X, k, clustering, smallest_error, sil, A = spectral.doit(
        graph_method, min_clusters=min_clusters, max_clusters=max_clusters,
        max_dist=max_dist, neighbors=neighbors, weight_function=weight_function
    )

    max_val = np.amax(A)

   # return calc_classification_error(X, y, clustering)
    #return scaled_X, calc_classification_error(X, y, clustering), k

    fig = plt.figure(figsize = (12,6), dpi= 80)

    """
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(range(min_clusters, max_clusters+1), smallest_error, color="#8a2be2")
    for i in range(min_clusters, max_clusters+1):
        ax.scatter(i, smallest_error[i-min_clusters], s=80, color="#8a2be2")
    ax.set_title("k-means++ error after spectral")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    bx = fig.add_subplot(2, 2, 2)
    bx.plot(range(min_clusters, max_clusters+1), sil, color="#8a2be2")
    for i in range(min_clusters, max_clusters+1):
        bx.scatter(i, sil[i-min_clusters], s=80, color="#8a2be2")
    bx.set_title("k-means++ silhouette coeffs after spectral")
    bx.xaxis.set_major_locator(MaxNLocator(integer=True))
    """

    cx = fig.add_subplot(1, 2, 1)
    for j in range(m):
        cx.scatter(scaled_X[j][0], scaled_X[j][1], color=colors[clustering[j]])
    cx.set_title(f"spectral for {k} clusters")
   
    dx = fig.add_subplot(1, 2, 2)
    for j in range(m):
        dx.scatter(scaled_X[j][0], scaled_X[j][1], color="#8a2be2")
    for i in range(m):
        for j in range(i+1, m):
            if A.item(i, j) > 0:
                if graph_method == 0 and A[i][j] < max_val/5:
                    continue
                print(f'plotting edge {i} <--> {j}')
                dx.plot([scaled_X[i][0], scaled_X[j][0]], [scaled_X[i][1], scaled_X[j][1]], color="#8a2be2", alpha=A[i][j]/max_val, linewidth=1)
    if graph_method == 0:
        dx.set_title(f'similarity graph - clique with weights')
    if graph_method == 1:
        dx.set_title(f"similarity graph for epsilon < {max_dist}")
    elif graph_method == 2:
        dx.set_title(f"similarity graph for {neighbors} nearest neighbors")

    fig.tight_layout()
    plt.show()

    return calc_classification_error(X, y, clustering)

i = sys.argv[1]
data = np.loadtxt(f"data/dane_2D_{i}.txt", dtype = float)
X, y = np.split(data, [2], axis = 1)

real_clusters = int(np.amax(y) - np.amin(y)) + 1

plt.rcParams['axes.grid'] = True

spectral_methods = [
    [0, 0, 0, lambda x : 1/x if x != 0 else 0],
    [0, 0, 0, lambda x : np.exp(-(x**2)/(0.01**2))],
    [0, 0, 0, lambda x : np.exp(-(x**2)/(0.1**2))],
    [0, 0, 0, lambda x : np.exp(-(x**2)/(0.5**2))],
    [0, 0, 0, lambda x : np.exp(-(x**2)/(1**2))],
    [0, 0, 0, lambda x : np.exp(-(x**2)/(10**2))],
    [1, 0.1, 0, lambda x : 1/x if x != 0 else 0],
    [1, 0.2, 0, lambda x : 1/x if x != 0 else 0],
    [1, 0.3, 0, lambda x : 1/x if x != 0 else 0],
    [1, 0.5, 0, lambda x : 1/x if x != 0 else 0],
    [1, 0.7, 0, lambda x : 1/x if x != 0 else 0],
    [2, 0, 3, lambda x : 1/x if x != 0 else 0],
    [2, 0, 5, lambda x : 1/x if x != 0 else 0],
    [2, 0, 10, lambda x : 1/x if x != 0 else 0],
    [2, 0, 3, lambda x : np.exp(-(x**2)/(0.5**2))],
    [2, 0, 5, lambda x : np.exp(-(x**2)/(0.5**2))],
    [2, 0, 10, lambda x : np.exp(-(x**2)/(0.5**2))]
]



