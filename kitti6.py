# draw 3d bbox on image using label data(gt box)

import numpy as np
import kitti_utils as kitti
import cv2

cv_box_colormap = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [1, 0, 255],
]

lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne\000010.bin"
label_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_label_2\training\label_2\000010.txt"
calib_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_calib\training\calib\000010.txt"
image_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_image_2\training\image_2\000010.png"

labels_feature_dict, class_ids, bbox2D_cam, bbox3D_cam = kitti.parse_label_file(label_file_path)
print(class_ids, labels_feature_dict["labels/obj_center_cam"], bbox3D_cam)
calib_feature_dict, matrix_proj_2, matrix_tr_velo_to_cam, R_cam_to_rect = kitti.parse_calib_file(calib_file_path)
image = cv2.imread(image_file_path)

for i in range(len(bbox3D_cam)):
    corners_3d_cam2 = kitti.compute_3d_box_cam2(bbox3D_cam[i])
    print(corners_3d_cam2)
    pts_2d = kitti.project_rect_to_image(matrix_proj_2, corners_3d_cam2.T)
    print(pts_2d)
    kitti.draw_box3d_on_image(image, pts_2d, cv_box_colormap[class_ids[i]], thickness=2)

cv2.imshow("Image", image)
cv2.waitKey()