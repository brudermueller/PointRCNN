import os
import numpy as np
import pickle
import torch

from lib.utils import custom_data_utils

""" 
Generate a Ground Truth Database pickle file for data augmentation.  
"""

#TODO: as soon as data augmentation will be needed

def generate_gt_database(self):
    #TODO: move to own GTDatabaseGenerator Class 
    self.gt_database = []
    for idx, sample in enumerate(self.current_samples):
        path = os.path.join(self.root, sample)
        data, labels, bboxes = custom_data_utils.load_h5(path, bbox=True)  
        pts_lidar = data[:,:3]
        if self.intensity_channel:
            pts_intensity = data[:, 4]

        #bbox defined as: (N, 7) [x, y, z, h, w, l, ry]
        gt_boxes3d = np.zeros((bboxes.__len__(), 7), dtype=np.float32)
        for k, bbox in enumerate(bboxes):
            centroid, h, w, l, angle = bbox[0:3], bbox[3], bbox[4], bbox[5], bbox[6] 
            gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = centroid, h, w, l, angle

        if gt_boxes3d.__len__() == 0:
            print('No gt object')
        continue