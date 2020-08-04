#!/usr/bin/env python
'''
    Data preparation for PointRCNN 
'''

import os 
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset

import lib.utils.custom_data_utils as data_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/hdf5/")

class CustomRCNNDataset(Dataset): 
    def __init__(self, root, num_points, split='train', mode='TRAIN', batch_size=10, normalize=False, 
                intensity_channel=True, shuffle=True, rcnn_training_roi_dir=None, 
                rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                rcnn_eval_feature_dir=None): 
        """
        :param root: directory path to the dataset
        :param split: 'train' or 'test'
        :param num_points: number of points to process for each pointcloud (needs to be the same)
        :param normalize: whether include the normalized coords in features (default: False)
        :param intensity_channel: whether to include the intensity value to xyz coordinates (default: True)
        :param shuffle: whether to shuffle the data (default: True)
        """
        self.root = root
        self.split = split 
        self.num_points = num_points # TODO: define number of points to train with per frame
        self.batch_size = batch_size
        self.normalize = normalize
        self.intensity_channel = intensity_channel 
        self.shuffle = shuffle
        self.classes = ('Background', 'Pedestrian')
        self.num_class = self.classes.__len__()

        self.sample_id_list = []
        

        # load data files 
        all_files = data_utils.get_data_files(self.root)

        # TODO: shuffle frames 
        # TODO: create batches 

        data_batchlist, label_batchlist = [], []                                                                                       
        for f in all_files:                                                                                                         
            data, labels, bbox = data_utils.load_h5(f, bbox=True)    
            #bbox: (N, 7) [x, y, z, h, w, l, ry]
            centroid, h, w, l, angle = bbox[0:3], bbox[3], bbox[4], bbox[5], bbox[6] 
            if not intensity_channel: 
                data = data[:,:3]                                                               
        
            # TODO: check if needed here 
            # reshaping to size (n, 1) instead of (n,) because pytorch wants it like that
            labels_reshaped = np.ones((labels.shape[0], 1), dtype=np.float) # pylint: disable=E1136
            labels_reshaped[:, 0] = labels[:]
            labels = labels_reshaped
        
            data_batchlist.append(data)                                                          
            label_batchlist.append(labels)    
        data_batches = np.concatenate(data_batchlist, 0)
        label_batches = np.concatenate(label_batchlist,0)
        
        self.data = data_batchlist
        self.labels = label_batchlist


        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

    def shuffle_data(self):
        """ Shuffle data and labels.
            
            Args: 
                data [numpy array]: B,N,... 
                label [numpy array]: B,... 
            Returns:
                shuffled data, label and shuffle indices
        """
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return self.data[idx, ...], self.labels[idx], idx


    def get_lidar(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        current_points = torch.from_numpy(self.data[index, pt_idxs].copy()).float()
        raise NotImplementedError

    def get_label(self, idx):
        current_labels = torch.from_numpy(self.labels[index, pt_idxs].copy()).long()
        raise NotImplementedError    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO change from random sampling to farthest point sampling 
        pt_idxs = np.arange(0, self.num_points)
        if self.shuffle: 
            np.random.shuffle(pt_idxs)

        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                if cfg.RCNN.ROI_SAMPLE_JIT:
                    return self.get_rcnn_sample_jit(index)
                else:
                    return self.get_rcnn_training_sample_batch(index)
            else:
                return self.get_proposal_from_file(index)
        else:
            raise NotImplementedError

        # return self.data, self.labels
    
    def get_rpn_sample(self, index):
        """ Creates input for region proposal network. 

        Args:
            index (int): The index of the point cloud instance. 
        """
        pts_lidar = self.get_lidar(index)
        pts_intensity = pts_lidar[:, 3]




if __name__ == '__main__':
    pass
    # dataset = ShapeNetDataset(
    #     root=opt.dataset,
    #     classification=False,
    #     class_choice=[opt.class_choice])
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=opt.batchSize,
    #     shuffle=True,
    #     num_workers=int(opt.workers))