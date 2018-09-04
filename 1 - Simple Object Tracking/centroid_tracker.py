
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

#OrderedDict is a special Python container that
#remembers the order entries were added



class CentroidTracker(object):
    
    def __init__(self, maxDisappeared=50):
        '''
        * maxDisappeared:
            max nº of frames lost until deregister
        '''
        
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
        
    def centroid_(x0,y0,x1,y1):
        '''
        * Functionality:
            Calculate the centroid coordinate of a given rectangle
        '''
        cx = int((x0+x1) / 2.0)
        cy = int((y0+y1) / 2.0)
        return tuple(cx,cy)
        
    
    def register(self, centroid):
        '''
        * Functionality:
            Identifies a new objects and register
            it with its centroid
        '''
        self.objects[self.nextObjectID] = centroid  # Register object 
        self.disappeared[self.nextObjectID] = 0     # Start disp counter
        self.nextObjectID += 1                      # Increase ID counter
        
    
    def deregister(self, objectID):
        '''
        * Functionality:
            Delete the object ID from the dicts
        * objetcID:
            ID of the object to deregister
        '''
        del self.objects[objectID]
        del self.disappeared[objectID]
        
        
    def update(self, rects: tuple):
        '''
        * rects:
            4-tuple: BB rectangles from an object detector
        '''
       
        # 1 - Functionality for disappearing BBs
        
        # If there are no detections:
        if len(rects) == 0:
            
            # Loop over the objects to increment their time being disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                
                # Check if the object has reach the maximum time lost
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                
            # Nothing else to do since there are no objects detected
            return self.objects
        
        
        # 2 - Initialize an array of centroids for the current frame
        
        inputCentroids = np.zeros((len(rects), 2), dtype='int')
        
        # Loop over the rectangles to calculate each centroid
        for (i, (x0,y0,xf,yf)) in enumerate(rects):
            inputCentroids[i] = self.centroid_(x0,y0,xf,yf)
            
        
        # 3 - Register new objetcs
        
        if len(self.objects) == 0: 
            
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
                
        # 4 - Update existing object
        
        else:
            
            # Grab the objects ids and centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Calculate distance of each pair of objects and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # 4.2 Match pairs of minimum distances
            
            # Sort the smallest distances 
            rows = D.min(axis=1).argsort() ## TODO: understand argsort 
            cols = D.argmin(axis=1)[rows]
            
            # Use the distances to associate objectsID
            usedRows = set()
            usedCols = set()
            
            for (row,col) in zip(rows,cols):
                
                # Ignore if we hace already examined
                if row in usedRows or col in usedCols: continue
            
                # Otherwise
                objectID = objectIDs[row]                       # Grab the ID
                self.objects[objectID] = inputCentroids[col]    # Set the new centroid (it has moved)
                self.disappeared[objectID] = 0                  # Reset counter (still on screen)
                
                # Remove them from next check
                usedRows.add(row)
                usedCols.add(col)
                
            # Cases we haven't examined yet
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[0])).difference(usedCols)
            
            # 5 - Verify if the extra object centroids have disappeared
            
            if D.shape[0] >= D.shape[1]:
                
                for row in unusedRows:
                    
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    # Check if the object has reach the maximum time lost
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            
            # 6 - Otherwise, it means we have new inputs to register
            
            else:
                
                for col in unusedCols:
                    self.register(inputCentroids[col])
                    
        return self.objects
                
            
        
import matplotlib.pyplot as plt
                    
np.random.seed(42)
objectCentroids = np.random.uniform(size=(2,2)) * 10
centroids = np.random.uniform(size=(3,2)) * 10
        
        
plt.figure()
plt.scatter(*zip(*objectCentroids), c='r')
plt.scatter(*zip(*centroids), c='b')
plt.show()
        

D = dist.cdist(objectCentroids, centroids)
D.min(axis=1)
rows = D.min(axis=1).argsort()
cols = D.argmin(axis=1)[rows]


