# draw 3d bbox on image using lidar coordinate(lidar 3d bbox)

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

# lidar 3d bbox (x,y,z,l,w,h,rz)
lidar_3bbox = np.array([ [ 5.4826984, -4.4219365, -0.9296398,  3.35,        1.65,        1.57,  -0.15079632],
                [12.081573,    2.3993492,  -0.8686094,   3.95,        1.7,         1.43,   2.952389  ],
                [23.789532,   -8.322558,   -0.48450655,  1.09,        0.72,        1.96,   2.962389  ],
                [16.782623,   -5.8402405,  -0.84653795,  3.24,        1.6,         1.51,  -0.13079633],
                [22.332664,   -6.859388,   -0.80930686,  4.1,        1.74,        1.45,  -0.18079633],
                [23.921867,    0.39140925, -0.81109357,  3.79,        1.68,        1.54,   2.932389  ],
                [29.351864,   -0.6278057,  -0.7701172,   3.35,        1.52,        1.49,   2.922389  ],
                [28.813488,   -7.8675747,  -0.8422416,   4.37,        1.65,        1.53,  -0.17079632],
                [43.13186,    -4.4860353,  -0.6518749,   3.48,        1.45,        1.64,   2.692389  ]]).astype(float)

labels_feature_dict, class_ids, bbox2D_cam, bbox3D_cam = kitti.parse_label_file(label_file_path)
print(class_ids, labels_feature_dict["labels/obj_center_cam"], bbox3D_cam)
calib_feature_dict, matrix_proj_2, matrix_tr_velo_to_cam, R_cam_to_rect = kitti.parse_calib_file(calib_file_path)
image = cv2.imread(image_file_path)

for i in range(len(lidar_3bbox)):
    
    open3d_bbox = kitti.create_open3d_bounding_box(lidar_3bbox[i])
    open3d_bbox_corner = np.asarray(open3d_bbox.get_box_points())
    print("open3d_bbox_corner",open3d_bbox_corner)
    
    ref_3bbox_cornder = kitti.project_velo_to_ref(matrix_tr_velo_to_cam, open3d_bbox_corner)
    print("ref_3bbox_cornder",ref_3bbox_cornder)
    
    rect_3bbox_cornder = kitti.project_ref_to_rect(R_cam_to_rect, ref_3bbox_cornder)
    print("rect_3bbox_cornder",rect_3bbox_cornder)

    pts_2d = kitti.project_rect_home_to_image(matrix_proj_2, rect_3bbox_cornder)
    print(pts_2d)
    kitti.draw_open3d_box3d_on_image(image, pts_2d, cv_box_colormap[class_ids[i]], thickness=2)
    

cv2.imshow("Image", image)
cv2.waitKey()