import numpy as np 
from hierarchical import HierarchicalClustering
import sys 

i = sys.argv[1]

data = np.loadtxt(f"data/dane_2D_{i}.txt", dtype = float)
X, y = np.split(data, [2], axis = 1)

hcluster = HierarchicalClustering(X, y)
hcluster.doit(0, 0, 2)