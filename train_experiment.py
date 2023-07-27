import torch
import utils
import os
display = utils.notebook_init()

baseline_train = "python train.py --img 640 --batch 32 --epochs 200 --data dataset_standard_train.yaml --weights yolov5s.pt --cache --single-cls --freeze 10"

# tune_train = "python train.py --img 640 --batch 32 --epochs 200 --data dataset_standard_train.yaml --weights yolov5s.pt --cache --single-cls --freeze 10 --evolve"


os.system(baseline_train)
