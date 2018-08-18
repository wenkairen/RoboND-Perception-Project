## Project: Perception Pick & Place

---
### Writeup / README

[image1]: ./images/e3_2.png
[image2]: ./images/e3_a.png
[image3]: ./images/e3_r.png
[image4]: ./images/12y.png
[image5]: ./images/1c.png
[image6]: ./images/1r.png
[image7]: ./images/22.png
[image8]: ./images/2r.png
[image9]: ./images/32.png
[image10]: ./images/3a.png
[image11]: ./images/3r.png

1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
the code is shown below:

```python 
    # Create the segmentation object
    seg = cloud_passthrough.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    # Extract inliers

    cloud_table = cloud_passthrough.extract(inliers, negative=False)

    # Extract outliers
    cloud_objects = cloud_passthrough.extract(inliers, negative=True)
```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
```python
    # Extract outliers
    cloud_objects = cloud_passthrough.extract(inliers, negative=True)

    ##TODO: Euclidean Clustering 
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(3000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
```

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained, and Object recognition implemented.

feature is extracted by flowing method:
```python
def compute_color_histograms(cloud, using_hsv=True):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # TODO: Compute histograms
    c1_hist, _ = np.histogram(channel_1_vals, bins = 128)
    c2_hist, _ = np.histogram(channel_2_vals, bins = 128)
    c3_hist, _ = np.histogram(channel_3_vals, bins = 128)

    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((c1_hist, c2_hist, c3_hist)).astype(np.float64)
    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features).astype(np.float64)
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    hist_x, _ = np.histogram(norm_x_vals, bins = 64)
    hist_y, _ = np.histogram(norm_y_vals, bins = 64)
    hist_z, _ = np.histogram(norm_z_vals, bins = 64)
    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((hist_x, hist_y, hist_z)).astype(np.float64)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features).astype(np.float64)

    return normed_features
```

then trained by SVM to save the model.sav file for testing, for each sample, we have captured 500 times for each object:

the training result and the confusion matrix is shown below, 




![alt text][image1]

![alt text][image2]

![alt text][image3]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

the testing result is shown below of each result for test1 , 2 and 3

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  

### Future disscuss:



