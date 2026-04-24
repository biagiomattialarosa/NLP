import os

from detectron2.data.datasets import register_coco_instances


root = os.getenv("DETECTRON2_DATASETS", "datasets")
path_cityperson = os.path.join(root, "cityperson/annotations/")
path_cityscape = os.path.join(root, "cityperson/data/images/")
register_coco_instances("cityperson_val", {}, path_cityperson +"val.json", path_cityscape)
register_coco_instances("cityperson_train", {}, path_cityperson +"train.json", path_cityscape)