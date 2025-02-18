import numpy as np
import open3d
import torch
import matplotlib

_SOURCE_FOLDER = "/data/datasets/KITTI"
_TRAINING_SUBFOLDERS = ["velodyne", "label_2", "image_2", "calib"]
_OBJ_TYPE_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2, "Truck":3}


def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle
        return angle


def get_bbox3d(obj_xyz_cam, rot_y, dimensions, tr_velo_to_cam, R_cam_to_rect):
    """returns 3D object location center (x, y, z) in lidar coordinate"""
    length = dimensions[2]
    width = dimensions[1]
    height = dimensions[0]
    rot_z = ry_to_rz(rot_y)

    # projection from camera coordinates to lidar coordinates
    obj_xyz_cam = np.vstack((obj_xyz_cam.reshape(3,1), [1]))
    rot_mat = np.linalg.inv(R_cam_to_rect @ tr_velo_to_cam) # 레이블데이터(중심좌표)가 camera Rectified 좌표계일때 사용
    #rot_mat = np.linalg.inv(tr_velo_to_cam) # 레이블데이터(중심좌표)가 카메라좌표계일때 사용
    obj_xyz_lidar = rot_mat @ obj_xyz_cam
    obj_x = obj_xyz_lidar[0][0]
    obj_y = obj_xyz_lidar[1][0]
    obj_z = obj_xyz_lidar[2][0]

    return np.array([obj_x, obj_y, obj_z, length, width, height, rot_z])


def parse_point_cloud(file_path):
    """(x, y, z, r), where (x, y, z) is the 3D coordinates
    and r is the reflectance value (referred as intensity in Waymo)
    """
    pc_data = np.fromfile(file_path, "<f4")
    pc_data = pc_data.reshape((-1, 4))
    pc_data_xyz = pc_data[:, :3].flatten().tolist()
    pc_data_reflectance = pc_data[:, -1].flatten().tolist()
    pc_data_features = {
        "LiDAR/point_cloud/num_valid_points": pc_data.shape[0],
        "LiDAR/point_cloud/xyz": pc_data_xyz,
        "LiDAR/point_cloud/obj_reflectance": pc_data_reflectance,
    }
    return pc_data_features, pc_data 


def parse_labels(file_path, tr_velo_to_cam, matrix_rectification):
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
             in camera coordinates (in meters)
    14    -> Rotation ry around Y-axis in camera coordinates [-pi..pi]

    Creates 3D bounding box label which contains
    [center (x, y, z), length, width, height, heading]
    """
    with open(file_path) as f:
        lines = f.readlines()

    class_ids_list = []
    truncations_list = []
    occlusions_list = []
    observation_angles_list = []
    bboxs_list = []
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
            bbox_coords = np.array(
                [obj_label[4], obj_label[5], obj_label[6], obj_label[7]]
            ).astype(float)
            dimension = np.array([obj_label[8], obj_label[9], obj_label[10]]).astype(float)
            center_cam = np.array([obj_label[11], obj_label[12], obj_label[13]]).astype(float) 
            center_cam[1] -= dimension[0]/2 # 레이블데이터의 중심좌표는 바운딩박스의 밑면의 중심좌표임 -> 카메라좌표계에서 y축을 수정해야함
            rotation = float(obj_label[14])

            bbox_3d_lidar = get_bbox3d(center_cam, rotation, dimension, tr_velo_to_cam, matrix_rectification)

            class_ids_list.append(class_id)
            truncations_list.append(truncated)
            occlusions_list.append(occluded)
            observation_angles_list.append(alpha)
            bboxs_list.append(bbox_coords)
            bboxs3D_list.append(bbox_3d_lidar)
            dimensions_list.append(dimension)
            centers_cam_list.append(center_cam)
            rotations_list.append(rotation)

    num_obj = len(lines)
    num_valid_labels = len(class_ids_list)
    #class_ids = np.array(class_ids_list, dtype=np.int64).flatten().tolist()
    class_ids = np.array(class_ids_list, dtype=np.int64)
    truncated = np.array(truncations_list, dtype=np.float32).flatten().tolist()
    occluded = np.array(occlusions_list, dtype=np.int64).flatten().tolist()
    alpha = np.array(observation_angles_list, dtype=np.float32).flatten().tolist()
    bbox = np.array(bboxs_list, dtype=np.float32).flatten().tolist()

    #bbox3D = np.array(bboxs3D_list, dtype=np.float32).flatten().tolist()
    bbox3D = np.array(bboxs3D_list, dtype=np.float32)
    dimensions = np.array(dimensions_list, dtype=np.float32).flatten().tolist()
    center_cam = np.array(centers_cam_list, dtype=np.float32).flatten().tolist()
    rotation_y = np.array(rotations_list, dtype=np.float32).flatten().tolist()

    labels_feature_dict = {
        "LiDAR/labels/num_valid_labels": num_valid_labels,
        "LiDAR/labels/num_obj": num_obj,
        "LiDAR/labels/class_ids": (class_ids),
        "LiDAR/labels/obj_truncated": (truncated),
        "LiDAR/labels/obj_occluded": (occluded),
        "LiDAR/labels/obj_alpha": (alpha),
        "LiDAR/labels/obj_bbox": (bbox),
        "LiDAR/labels/box_3d": bbox3D,
        "LiDAR/labels/obj_dimensions": (dimensions),
        "LiDAR/labels/obj_center_cam": (center_cam),
        "LiDAR/labels/obj_rotation_y": (rotation_y),
    }

    return labels_feature_dict, class_ids, bbox3D


def parse_calib(file_path):
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
        "calib/matrix_proj_0": (matrix_proj_0.flatten().tolist()),
        "calib/matrix_proj_1": (matrix_proj_1.flatten().tolist()),
        "calib/matrix_proj_2": (matrix_proj_2.flatten().tolist()),
        "calib/matrix_proj_3": (matrix_proj_3.flatten().tolist()),
        "calib/matrix_rectification": (R_cam_to_rect.flatten().tolist()),
        "calib/matrix_tr_velo_to_cam": matrix_tr_velo_to_cam.flatten().tolist(),
        "calib/matrix_tr_imu_to_velo": (matrix_tr_imu_to_velo.flatten().tolist()),
    }

    return calib_feature_dict, matrix_tr_velo_to_cam, R_cam_to_rect


box_colormap = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
]

class_labels = ["Car", "Pedestrian", "Cyclist"]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    
    vis.create_window(window_name='kitti',width=960, height=540)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        line_set_list = create_line_set_list(gt_boxes, (0, 0, 1),ref_labels,ref_scores)

    #if ref_boxes is not None:
    #    create_line_set_list(ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    
    for i in range(len(line_set_list)):
        vis.add_geometry(line_set_list[i]) 

    # set zoom, front, up, and lookat
    vis.get_view_control().set_zoom(0.2)
    vis.get_view_control().set_front([0, 0, 1])
    vis.get_view_control().set_up([1, 0, 0])
    vis.get_view_control().set_lookat([0, 0, 0])
    
    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set


def create_line_set_list(gt_boxes, color=[0, 1, 0], ref_labels=None, score=None):
    print('box_no:', gt_boxes.shape[0],ref_labels )
    #print('bbox:',gt_boxes)
    
    line_set_list = []
    for i in range(gt_boxes.shape[0]):
        
        line_set = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        
        line_set_list.append(line_set)
        
        """
        if ref_labels is not None:
             corners = box3d.get_box_points()
             vis.add_3d_label(corners[5], class_labels[ref_labels[i]])
        if score is not None:
             corners = box3d.get_box_points()
             vis.add_3d_label(corners[5], '%.2f' % score[i])
        """
    return line_set_list

def create_open3d_bounding_box(gt_boxes):
    """
    3D 바운딩 박스를 Open3D 객체로 변환하는 함수
    """
    center = gt_boxes[:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6]])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    open3d_bbox = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    
    return open3d_bbox

def create_open3d_bbox_list(gt_boxes, ref_labels=None, score=None):
    print('box_no:', gt_boxes.shape[0],ref_labels )
    #print('bbox:',gt_boxes)
    
    open3d_bbox_list = []
    for i in range(gt_boxes.shape[0]):
        
        open3d_bbox = create_open3d_bounding_box(gt_boxes[i])
        if ref_labels is None:
            open3d_bbox.color = [0, 1, 0] 
        else:
            open3d_bbox.color = box_colormap[ref_labels[i]]

        open3d_bbox_list.append(open3d_bbox)        
       
    return open3d_bbox_list