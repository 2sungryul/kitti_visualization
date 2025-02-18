import numpy as np
import open3d as o3d

lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne\000000.bin"

"""
bin format
(x, y, z, r), where (x, y, z) is the 3D coordinates
and r is the reflectance value (referred as intensity in Waymo)
"""
# read point cloud file(bin format)
pc_data = np.fromfile(lidar_file_path, "<f4")
pc_data = pc_data.reshape((-1, 4))

# create open3d tensor
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)))

# create open3d window
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='kitti',width=960, height=540)

# set background_color and point_size
vis.get_render_option().background_color = np.asarray([0,0,0]).astype(float)
vis.get_render_option().point_size = 0.5

# add lidar axis
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)

# add point cloud
vis.add_geometry(pcd)

# set zoom, front, up, and lookat
vis.get_view_control().set_zoom(0.1)
vis.get_view_control().set_front([0, 0, 1])
vis.get_view_control().set_up([1, 0, 0])
vis.get_view_control().set_lookat([0, 0, 0])

vis.run()
vis.destroy_window()

