import numpy as np
import open3d as o3d

lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne\000000.bin"
image_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_image_2\training\image_2\000000.txt"

"""(x, y, z, r), where (x, y, z) is the 3D coordinates
    and r is the reflectance value (referred as intensity in Waymo)
"""
pc_data = np.fromfile(lidar_file_path, "<f4")
pc_data = pc_data.reshape((-1, 4))
pc_data_xyz = pc_data[:, :3].flatten().tolist()
pc_data_reflectance = pc_data[:, -1].flatten().tolist()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
#pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)))
coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3, origin=np.array([0.0, 0.0, 0.0]))
o3d.visualization.draw_geometries([pcd,coord],
                                  window_name='kitti',
                                  width=960, height=540,
                                  point_show_normal = True,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

