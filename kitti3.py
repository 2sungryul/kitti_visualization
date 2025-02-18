import kitti_utils as kitti

lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne\000010.bin"
label_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_label_2\training\label_2\000010.txt"
calib_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_calib\training\calib\000010.txt"

_, pc_data = kitti.parse_point_cloud(lidar_file_path)
_, matrix_tr_velo_to_cam, R_cam_to_rect = kitti.parse_calib(calib_file_path)
_, class_ids, bbox3D = kitti.parse_labels(label_file_path, matrix_tr_velo_to_cam, R_cam_to_rect)
print("bbox3D", bbox3D)
kitti.draw_scenes(pc_data, bbox3D, ref_labels=class_ids)

