# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# lidar box animation example

import numpy as np
import open3d as o3d
import threading
import time
import os
import kitti_utils as kitti

lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne"
label_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_label_2\training\label_2"
calib_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_calib\training\calib"

CLOUD_NAME = "points"
FRAME_NUM = 60

class MultiWinApp:

    def __init__(self):
        self.is_done = False
        self.cloud = None
        self.main_vis = None
        self.frame_index = 0
        self.first = False
        self.bbox_num = 0
        #self.n_snapshots = 0
        #self.snapshot_pos = None        

    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer("KITTI", 1280, 720)
        self.main_vis.reset_camera_to_default()
        self.main_vis.setup_camera(60, [0, 0, 0], [-15, 0, 10], [5, 0, 10]) # center, eye, up
        
        self.main_vis.set_background(np.array([0, 0, 0, 0]), None)
        self.main_vis.show_skybox(False)
        self.main_vis.point_size = 1
        self.main_vis.show_settings = True
                
        self.main_vis.set_on_close(self.on_main_window_closing)
        app.add_window(self.main_vis)
        threading.Thread(target=self.update_thread).start()

        #self.main_vis.add_action("Take snapshot in new window", self.on_snapshot)
        #self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        app.run()
    
    
    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        
        # parse bin, calib, and label files
        point_cloud_frames = load_point_cloud_frames(lidar_file_path)
        print(len(point_cloud_frames), len(point_cloud_frames[0]),len(point_cloud_frames[0][0]))
        matrix_tr_velo_to_cam_list, R_cam_to_rect_list = load_calib_list(calib_file_path)
        print(len(matrix_tr_velo_to_cam_list), len(matrix_tr_velo_to_cam_list[0]),len(matrix_tr_velo_to_cam_list[0][0]))
        print(len(R_cam_to_rect_list), len(R_cam_to_rect_list[0]),len(R_cam_to_rect_list[0][0]))
        class_ids_list, bbox3D_list = load_label_list(label_file_path, matrix_tr_velo_to_cam_list, R_cam_to_rect_list)
        print(len(class_ids_list), len(class_ids_list[0]))
        print(len(bbox3D_list), len(bbox3D_list[0]),len(bbox3D_list[0][0]), bbox3D_list[0][0])
   
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #self.main_vis.add_geometry("axis", axis_pcd)
        
        # Initialize point cloud geometry
        point_cloud = o3d.geometry.PointCloud()
        #point_cloud.points = o3d.utility.Vector3dVector(point_cloud_frames[0][:, :3])
        #point_cloud.colors = o3d.utility.Vector3dVector(np.ones((point_cloud_frames[0].shape[0], 3)))
        #self.main_vis.add_geometry("pc", point_cloud)
        
        #line_set_list = kitti.draw_box(bbox3D_list[0], ref_labels=class_ids_list[0])
        #self.main_vis.add_geometry("bbox", line_set_list[0])
   
        while not self.is_done:
                       
            time.sleep(1)
                   
            def update_cloud():
                print("frame_index:",self.frame_index )
                if self.first:
                    self.main_vis.remove_geometry("axis")
                    self.main_vis.remove_geometry("pc")
                    for i in range(self.bbox_num):
                        self.main_vis.remove_geometry(f"bbox{i}")
                                
                axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                self.main_vis.add_geometry("axis", axis_pcd)

                # Update point cloud with new data
                point_cloud.points = o3d.utility.Vector3dVector(point_cloud_frames[self.frame_index][:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(np.ones((point_cloud_frames[self.frame_index].shape[0], 3)))
                self.main_vis.add_geometry("pc", point_cloud)
                #print(len(bbox3D_list), len(bbox3D_list[0]),len(bbox3D_list[0][0]), bbox3D_list[0][0])                
                """
                line_set_list = kitti.create_line_set_list(bbox3D_list[self.frame_index], ref_labels=class_ids_list[self.frame_index])
                #print(len(line_set_list),line_set_list[0])
                
                self.bbox_num = len(line_set_list)
                for i in range(self.bbox_num):
                    self.main_vis.add_geometry(f"bbox{i}", line_set_list[i])
                """                                                
                open3d_bbox_list = kitti.create_open3d_bbox_list(bbox3D_list[self.frame_index], class_ids_list[self.frame_index])
                self.bbox_num = len(open3d_bbox_list)
                for i in range(self.bbox_num):
                    self.main_vis.add_geometry(f"bbox{i}", open3d_bbox_list[i])

                # Move to the next frame
                self.frame_index = (self.frame_index + 1) % FRAME_NUM
                self.first = True
                # save screen image to jpg                
                #self.main_vis.capture_screen_image("pc_%06d.jpg" % frame_index)
                
            o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, update_cloud)            

            if self.is_done:  # might have changed while sleeping
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


def main():
    MultiWinApp().run()

if __name__ == "__main__":
    main()