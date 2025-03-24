# Running a pretrained model for 3d object detection using open3d-ml library
# 3d od model : PointPillars

import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d.ml.vis import Visualizer
from tqdm import tqdm

def filter_detections(detections, min_conf = 0.5):
	good_detections = []
	for detection in detections:
		if detection.confidence >= min_conf:
			good_detections.append(detection)
	return good_detections


cfg_file = "./pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/KITTI/data_object_velodyne"
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="cuda", **cfg.pipeline)

# download the weights.
ckpt_folder = "./"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("training")

# Prepare the visualizer 
#vis = Visualizer()
vis = ml3d.vis.Visualizer()

# Variable to accumulate the predictions
data_list = []
# Let's detect objects in the first few point clouds of the Kitti set
for idx in tqdm(range(1)):
    # Get one test point cloud from the SemanticKitti dataset
    data = test_split.get_data(idx)
    #print(data.__class__)
    print('data:',type(data))
    #print(len(data))
    print(data.keys())
    
    # Run the inference
    result = pipeline.run_inference(data)[0]
    #print(result.__class__)
    #print('result:',type(result))
    #print(len(result))
    #print(result)
    #print('result[0]:',type(result[0]))

    # Filter out results with low confidence
    result = filter_detections(result)
	#print(result.__class__)
    print('result:',type(result))
    print(len(result))
    print(result)
    print('result[0]:',type(result[0]))
    # Prepare a dictionary usable by the visulization tool
    pred = {
    "name": 'KITTI' + '_' + str(idx),
    'points': data['point'],
    'bounding_boxes': result
    }
    # Append the data to the list    
    data_list.append(pred)
   
# Visualize the results
vis.visualize(data_list, None, bounding_boxes=None)   