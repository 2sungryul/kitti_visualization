import open3d as o3d

print("Testing IO for point cloud ...")
ply = o3d.io.read_point_cloud("copy_of_fragment.ply")
print(ply)
o3d.visualization.draw_geometries([ply],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


