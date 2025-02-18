# draw 3d bbox on image using label data(gt box)

import numpy as np
import cv2

_SOURCE_FOLDER = "/data/datasets/KITTI"
_TRAINING_SUBFOLDERS = ["velodyne", "label_2", "image_2", "calib"]
_OBJ_TYPE_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2, "Truck":3}

box_colormap = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [1, 0, 255],
]

def parse_label_file(file_path):
    """Extracts relevant information from label file
    0     -> Object type
    1     -> from 0 (non-truncated) to 1 (truncated), where truncated
             refers to the object leaving image boundaries
    2     -> 0 = fully visible, 1 = partly occluded,
             2 = largely occluded, 3 = unknown
    3     -> Observation angle of object [-pi..pi]
    4:7   -> 2D bounding box of object in the image,
             contains left, top, right, bottom pixel coordinates
    8:10  -> 3D object dimensions: height, width, length (in meters)
    11:13 -> The bottom center location x, y, z of the 3D object
             in camera coordinates (in meters) -> rectified coordinate
    14    -> Rotation ry around Y-axis in camera coordinates [-pi..pi]

    Creates 3D bounding box label which contains [h, w, l, x, y, z, ry]
    """
    with open(file_path) as f:
        lines = f.readlines()

    class_ids_list = []
    truncations_list = []
    occlusions_list = []
    observation_angles_list = []
    bboxs2D_list = []
    bboxs3D_list = []
    dimensions_list = []
    centers_cam_list = []
    rotations_list = []

    for line in lines:
        obj_label = line.strip().split()
        obj_type = obj_label[0]

        if obj_type in _OBJ_TYPE_MAP:
            class_id = _OBJ_TYPE_MAP[obj_type]
            truncated = float(obj_label[1])
            occluded = int(obj_label[2])
            alpha = float(obj_label[3])
            bbox2d_cam = np.array( [obj_label[4], obj_label[5], obj_label[6], obj_label[7]] ).astype(float)
            dimension = np.array([obj_label[8], obj_label[9], obj_label[10]]).astype(float)
            center_cam = np.array([obj_label[11], obj_label[12], obj_label[13]]).astype(float) 
            center_cam[1] -= dimension[0]/2 # 레이블데이터의 중심좌표는 바운딩박스의 밑면의 중심좌표임 -> 카메라좌표계에서 y축을 수정해야함
            rotation = float(obj_label[14])

            #bbox3d_cam = np.array([obj_label[8], obj_label[9], obj_label[10],obj_label[11], obj_label[12]-obj_label[8]/2, obj_label[13],obj_label[14]]).astype(float)
            bbox3d_cam = np.array([obj_label[8], obj_label[9], obj_label[10],obj_label[11], obj_label[12], obj_label[13],obj_label[14]]).astype(float)
            bbox3d_cam[4] -= bbox3d_cam[0]/2 # 레이블데이터의 중심좌표는 바운딩박스의 밑면의 중심좌표임 -> 카메라좌표계에서 y축을 수정해야함

            class_ids_list.append(class_id)
            truncations_list.append(truncated)
            occlusions_list.append(occluded)
            observation_angles_list.append(alpha)
            bboxs2D_list.append(bbox2d_cam)
            bboxs3D_list.append(bbox3d_cam)
            dimensions_list.append(dimension)
            centers_cam_list.append(center_cam)
            rotations_list.append(rotation)

    num_obj = len(lines)
    num_valid_labels = len(class_ids_list)
    class_ids = np.array(class_ids_list, dtype=np.int64)
    truncated = np.array(truncations_list, dtype=np.float32)
    occluded = np.array(occlusions_list, dtype=np.int64)
    alpha = np.array(observation_angles_list, dtype=np.float32)
    bbox2D_cam = np.array(bboxs2D_list, dtype=np.float32)

    bbox3D_cam = np.array(bboxs3D_list, dtype=np.float32)
    dimensions = np.array(dimensions_list, dtype=np.float32)
    center_cam = np.array(centers_cam_list, dtype=np.float32)
    rotation_y = np.array(rotations_list, dtype=np.float32)

    labels_feature_dict = {
        "labels/num_valid_labels": num_valid_labels,
        "labels/num_obj": num_obj,
        "labels/class_ids": (class_ids),
        "labels/obj_truncated": (truncated),
        "labels/obj_occluded": (occluded),
        "labels/obj_alpha": (alpha),
        "labels/obj_bbox": bbox2D_cam,
        "labels/box_3d": bbox3D_cam,
        "labels/obj_dimensions": (dimensions),
        "labels/obj_center_cam": (center_cam),
        "labels/obj_rotation_y": (rotation_y),
    }

    return labels_feature_dict, class_ids, bbox2D_cam, bbox3D_cam

def draw_box3d_on_image(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA )

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA )

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA )
    return image
    
def parse_calib_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    matrix_proj_0 = np.array(lines[0].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_1 = np.array(lines[1].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_2 = np.array(lines[2].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_3 = np.array(lines[3].strip().split(":")[1].split(), dtype=np.float32)
    matrix_rectification = np.array(lines[4].strip().split(":")[1].split(), dtype=np.float32)
    matrix_tr_velo_to_cam = np.array(lines[5].strip().split(":")[1].split(), dtype=np.float32)
    matrix_tr_imu_to_velo = np.array(lines[6].strip().split(":")[1].split(), dtype=np.float32)

    matrix_tr_velo_to_cam = np.vstack((matrix_tr_velo_to_cam.reshape(3,4), [0., 0., 0., 1.]))
    matrix_proj_2 = np.vstack((matrix_proj_2.reshape(3, 4), [0., 0., 0., 0.]))
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = matrix_rectification.reshape(3, 3)

    calib_feature_dict = {
        "calib/matrix_proj_0": matrix_proj_0,
        "calib/matrix_proj_1": matrix_proj_1,
        "calib/matrix_proj_2": matrix_proj_2,
        "calib/matrix_proj_3": matrix_proj_3,
        "calib/matrix_rectification": R_cam_to_rect,
        "calib/matrix_tr_velo_to_cam": matrix_tr_velo_to_cam,
        "calib/matrix_tr_imu_to_velo": matrix_tr_imu_to_velo,
    }

    return calib_feature_dict, matrix_proj_2, matrix_tr_velo_to_cam, R_cam_to_rect

def compute_3d_box_cam2(bbox3D_cam):
    """
    Return : 3xn in cam2 coordinate
    """
    h = bbox3D_cam[0]
    w = bbox3D_cam[1]
    l = bbox3D_cam[2]
    x = bbox3D_cam[3]
    y = bbox3D_cam[4]
    z = bbox3D_cam[5]
    yaw = bbox3D_cam[6]

    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    #y_corners = [0,0,0,0,-h,-h,-h,-h]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2
 
def project_rect_to_image(P, pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(P)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

lidar_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_velodyne\training\velodyne\000010.bin"
label_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_label_2\training\label_2\000010.txt"
calib_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_calib\training\calib\000010.txt"
image_file_path = r"D:\Users\2sungryul\Dropbox\Work\Dataset\KITTI\data_object_image_2\training\image_2\000010.png"

labels_feature_dict, class_ids, bbox2D_cam, bbox3D_cam = parse_label_file(label_file_path)
print(class_ids, labels_feature_dict["labels/obj_center_cam"], bbox3D_cam)
calib_feature_dict, matrix_proj_2, matrix_tr_velo_to_cam, R_cam_to_rect = parse_calib_file(calib_file_path)
image = cv2.imread(image_file_path)

for i in range(len(bbox3D_cam)):
    corners_3d_cam2 = compute_3d_box_cam2(bbox3D_cam[i])
    print(corners_3d_cam2)
    pts_2d = project_rect_to_image(matrix_proj_2, corners_3d_cam2.T)
    print(pts_2d)
    draw_box3d_on_image(image, pts_2d, box_colormap[class_ids[i]], thickness=2)

cv2.imshow("Image", image)
cv2.waitKey()