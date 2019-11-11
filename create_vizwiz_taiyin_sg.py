from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os

import sys
sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.spice.spice import Spice

train_path = '/home/yz9244/Up-Down-Captioner/bottom-up-attention/data/VizWiz/annotations_taiyin_full/VizWiz_Captions_v1_train.json'
val_path = '/home/yz9244/Up-Down-Captioner/bottom-up-attention/data/VizWiz/annotations_taiyin_full/VizWiz_Captions_v1_val.json'
test_path = '/home/yz9244/Up-Down-Captioner/bottom-up-attention/data/VizWiz/annotations_taiyin_full/VizWiz_Captions_v1_test.json'

coco_train = COCO(train_path)
coco_val = COCO(val_path)
coco_test = COCO(test_path)

coco_use = coco_val

image_ids = coco_use.getImgIds()
gts = {}
res = {}
for img_id in image_ids:
	gts[img_id] = []
	data_temp = coco_use.imgToAnns[img_id]
	for dt in data_temp:
		gts[img_id].append(dt['caption'])
	res[img_id] = []
	res[img_id].append(gts[img_id][0])

scorer = Spice()
score, scores = scorer.compute_score(gts, res)
