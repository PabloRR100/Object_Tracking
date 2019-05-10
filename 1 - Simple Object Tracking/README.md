
# Centroid Tracking Algorithm


THIS POST IS BASED ON [ADRIEN ROSEBROOK SITE][REF]


The centroid tracking algorithm assumes that we are passing in a set of bounding box (x, y)-coordinates for each detected object in every single frame.  

These bounding boxes can be produced by any type of object detector you would like that they are computed for every frame in the video:  
- Region Proposal Based CNNs
- Single Shot Detector
- YOLO 

### First Frame

For the first frame we will detect the objects.  
Because it is the first time we see them, we are sure they are different entities.  
Therefore, we assign a different ID to each of them.  
![][01_ids]

### Second Frame

We are now simulating a second time frame where we will encounter new location for the points that are representing potential centroids.  

The goal is to identify which point belongs to the previous centroids that we had. To do that, we will assign each of them to the closest previous centroid that they have.  
![][02_new_ids]

This [small post][dist_post] shows how the distances calculation is done.


## Different number of centroids

We need to be aware that we can contemplate the scenario where an object that has been tracking dissapears (either going out from the window or because something is between the object and the camera). Also, new objects to be tracked could appear all of a sudden.  

How do we handle these cases?  

### Register new objects

If there are more input detections than existing objects being tracked, we need to register the new object. “Registering” simply means that we are adding the new object to our list of tracked objects by:

- 1: Assigning it a new object ID
- 2: Storing the centroid of the bounding box coordinates for that object

The results will therefore looks:
![][03_regiter_id]


### Deregister missing objects

We will deregister old objects when they cannot be matched to any existing objects for a total of N subsequent frames.




[//]: # (Images of Console)

[01_ids]: https://www.pyimagesearch.com/wp-content/uploads/2018/07/simple_object_tracking_step1.png   

[02_new_ids]: https://www.pyimagesearch.com/wp-content/uploads/2018/07/simple_object_tracking_step2.png

[03_regiter_id]: https://www.pyimagesearch.com/wp-content/uploads/2018/07/simple_object_tracking_step4.png

[//]: # (Internal Repo Links)
[dist_post]: https://github.com/PabloRR100/Object_Tracking/blob/master/1%20-%20Simple%20Object%20Tracking/Centroid%20Distances%20Calculations/Centroid%20Distances%20Calculations.md   


[REF]: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/



```python
 
```
