        
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
                    
np.random.seed(42)
objectCentroids = np.random.uniform(size=(3,2)) * 10
centroids = np.random.uniform(size=(5,2)) * 10
        
        
plt.figure()
plt.scatter(*zip(*objectCentroids), c='r')
plt.scatter(*zip(*centroids), c='b')
plt.show()
        

D = dist.cdist(objectCentroids, centroids)
D.min(axis=1)
rows = D.min(axis=1).argsort()
cols = D.argmin(axis=1)[rows]

rows, cols

