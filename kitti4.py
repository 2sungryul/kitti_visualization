import open3d as o3d
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import kitti_utils as kitti
FRAME_NUM = 10

def animate_point_clouds(point_cloud_frames, bbox3D_list, class_ids_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='kitti',width=960, height=540)
    
    # Set background color to black
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0, 0, 0])
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    # Initialize point cloud geometry
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_frames[0][:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(np.ones((point_cloud_frames[0].shape[0], 3)))
    
    vis.add_geometry(point_cloud)
    _,line_set = kitti.draw_box(vis, bbox3D_list[0], ref_labels=class_ids_list[0])
    vis.add_geometry(line_set)
    
    # set zoom, front, up, and lookat
    vis.get_view_control().set_zoom(0.05)
    vis.get_view_control().set_front([-2, 0, 1])
    vis.get_view_control().set_up([1, 0, 1])
    vis.get_view_control().set_lookat([0, 0, 0])
    
    vis.poll_events()
    vis.update_renderer()

    frame_index = 0
    last_update_time = time.time()
    update_interval = 1  # Time in seconds between frame updates
    
    while True:
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            #vis.clear_geometries()
            #vis.reset_view_point()
           
            vis.add_geometry(axis_pcd)

            # Update point cloud with new data
            point_cloud.points = o3d.utility.Vector3dVector(point_cloud_frames[frame_index][:, :3])
            point_cloud.colors = o3d.utility.Vector3dVector(np.ones((point_cloud_frames[frame_index].shape[0], 3)))
            #vis.update_geometry(point_cloud)
            vis.add_geometry(point_cloud)
            _, line_set = kitti.draw_box(vis, bbox3D_list[frame_index], ref_labels=class_ids_list[frame_index])
            print(line_set)
            #vis.update_geometry(line_set)
            vis.add_geometry(line_set)
            
            
            vis.get_view_control().set_zoom(0.05)
            vis.get_view_control().set_front([-2, 0, 1])
            vis.get_view_control().set_up([1, 0, 1])
            vis.get_view_control().set_lookat([0, 0, 0])
    
            # Move to the next frame
            frame_index = (frame_index + 1) % len(point_cloud_frames)
            last_update_time = current_time
            vis.capture_screen_image("pc_%06d.jpg" % frame_index)
        vis.poll_events()
        vis.update_renderer()
        print(frame_index)
        if not vis.poll_events():
            break
    vis.destroy_window()

def animate_point_clouds2(point_cloud_frames, bbox3D_list, class_ids_list):
    
    frame_index = 0
       
    while True:
        print(frame_index)
        vis = o3d.visualization.Visualizer()    
        print(len(class_ids_list), len(class_ids_list[frame_index]))
        print(len(bbox3D_list), len(bbox3D_list[frame_index]),len(bbox3D_list[frame_index][0]))
        
        vis.create_window(window_name='kitti',width=960, height=540)
        
        # Set background color to black
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([0, 0, 0])
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        # Initialize point cloud geometry
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_frames[frame_index][:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(np.ones((point_cloud_frames[frame_index].shape[0], 3)))
        vis.add_geometry(point_cloud)
        
        _,line_set = kitti.draw_box(vis, bbox3D_list[frame_index], ref_labels=class_ids_list[frame_index])
        vis.add_geometry(line_set)
        
        # set zoom, front, up, and lookat
        vis.get_view_control().set_zoom(0.05)
        vis.get_view_control().set_front([-2, 0, 1])
        vis.get_view_control().set_up([1, 0, 1])
        vis.get_view_control().set_lookat([0, 0, 0])
        
        vis.run()
        #vis.poll_events()
        #vis.update_renderer()
        
        # Move to the next frame
        #frame_index = (frame_index + 1) % len(point_cloud_frames)
        frame_index = (frame_index + 1)
        vis.capture_screen_image("pc_%06d.jpg" % frame_index)
        #vis.remove_geometry()
        vis.destroy_window()
        vis =0

        
        if frame_index >= FRAME_NUM:
            break
    

def load_point_cloud_frames(directory):
    point_clouds = []
    index = 0;
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.bin'):  # Check for bin
            file_path = os.path.join(directory, filename)
            print(file_path)
            pc_data = np.fromfile(file_path, "<f4")
            pc_data = pc_data.reshape((-1, 4))
            point_clouds.append(pc_data)
            
            index = index + 1;
            
            if index >= FRAME_NUM:
                break
                     
    return point_clouds

def load_label_list(directory,matrix_tr_velo_to_cam_list, R_cam_to_rect_list):
    class_ids_list = []
    bbox3D_list = []
    index = 0;
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):  # Check for bin
            file_path = os.path.join(directory, filename)
            print(file_path)
            
            _, class_ids, bbox3D = kitti.parse_labels(file_path, matrix_tr_velo_to_cam_list[index], R_cam_to_rect_list[index])
                        
            class_ids_list.append(class_ids)
            bbox3D_list.append(bbox3D)
                        
            index = index + 1;
            if index >= FRAME_NUM:
                break
                     
    return class_ids_list, bbox3D_list

def load_calib_list(directory):
    matrix_tr_velo_to_cam_list = []
    R_cam_to_rect_list = []
    index = 0;
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):  # Check for bin
            file_path = os.path.join(directory, filename)
            print(file_path)
            
            _, matrix_tr_velo_to_cam, R_cam_to_rect = kitti.parse_calib(file_path)
               
            matrix_tr_velo_to_cam_list.append(matrix_tr_velo_to_cam)
            R_cam_to_rect_list.append(R_cam_to_rect)
            
            index = index + 1;
            if index >= FRAME_NUM:
                break
                     
    return matrix_tr_velo_to_cam_list, R_cam_to_rect_list

if __name__ == '__main__':

    lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne"
    label_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_label_2\training\label_2"
    calib_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_calib\training\calib"

    point_cloud_frames = load_point_cloud_frames(lidar_file_path)
    print(len(point_cloud_frames), len(point_cloud_frames[0]),len(point_cloud_frames[0][0]))
    matrix_tr_velo_to_cam_list, R_cam_to_rect_list = load_calib_list(calib_file_path)
    print(len(matrix_tr_velo_to_cam_list), len(matrix_tr_velo_to_cam_list[0]),len(matrix_tr_velo_to_cam_list[0][0]))
    print(len(R_cam_to_rect_list), len(R_cam_to_rect_list[0]),len(R_cam_to_rect_list[0][0]))
    class_ids_list, bbox3D_list = load_label_list(label_file_path, matrix_tr_velo_to_cam_list, R_cam_to_rect_list)
    print(len(class_ids_list), len(class_ids_list[0]))
    print(len(bbox3D_list), len(bbox3D_list[0]),len(bbox3D_list[0][0]))
    
    animate_point_clouds(point_cloud_frames,bbox3D_list,class_ids_list)
    #animate_point_clouds2(point_cloud_frames,bbox3D_list,class_ids_list)