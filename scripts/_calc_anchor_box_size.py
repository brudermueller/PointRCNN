#!/usr/bin/env python

'''
    Script to calculate the average anchor box size from training data, which is used as an input for the RPN network. 
'''
import _init_path

import os 
import numpy as np 
from lib.utils import custom_data_utils as data_utils


def calc_mean_anchor_box_size(): 
    print(os.getcwd())
    root = '../data/custom_data'
    train_dir = os.path.join(root, 'train.txt')
    train_samples = data_utils.get_data_files(train_dir)
    height_vals = []
    width_vals = []
    length_vals = []
    print('Loading and evaluating {} files from {}'.format(len(train_samples), train_dir))
    for file_name in train_samples: 
        lidar_file = os.path.join(root, file_name)
        _, _, bbox = data_utils.load_h5(lidar_file, bbox=True)
        height_vals.append(float(bbox[3]))
        width_vals.append(float(bbox[4]))
        length_vals.append(float(bbox[5]))
    return(np.mean(height_vals), np.mean(width_vals),np.mean(length_vals))


if __name__=='__main__': 
    mean_height, mean_width, mean_length = calc_mean_anchor_box_size()
    print('Mean anchor-box-size (h,w,l): {}, {}, {}'.format(mean_height, mean_length, mean_width))
