import open3d as o3d

print("Testing IO for point cloud ...")
sample_ply_data = o3d.data.PLYPointCloud()
ply = o3d.io.read_point_cloud(sample_ply_data.path)
print(ply)
o3d.io.write_point_cloud("copy_of_fragment.ply", ply)
