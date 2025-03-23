# kitti visualization(point cloud with bbox label) using open3d-ml library

import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.KITTI(dataset_path="/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/KITTI/data_object_velodyne")

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('all')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0).keys())
print(all_split.get_data(0)['point'].shape)
print(all_split.get_data(0)['bounding_boxes'].__class__)
print(all_split.get_data(0)['bounding_boxes'])
print(all_split.get_data(0)['bounding_boxes'][0].__class__)
print(all_split.get_data(0)['bounding_boxes'][0])
print(all_split.get_data(0)['bounding_boxes'][0].center)
print(all_split.get_data(0)['bounding_boxes'][0].size)
print(all_split.get_data(0)['bounding_boxes'][0].yaw)
print(all_split.get_data(0)['bounding_boxes'][0].label_class)
print(all_split.get_data(0)['bounding_boxes'][0].confidence)
print(all_split.get_data(0)['bounding_boxes'][0].world_cam)
print(all_split.get_data(0)['bounding_boxes'][0].cam_img)

# show the first 400 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, "training", indices=range(50))