#!/usr/bin/env python
'''
    Prepare data from custom dataset for PointRCNN architecture
'''

import os 
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset

import lib.utils.custom_data_utils as data_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/custom_data/")

def shuffle_data(data, labels):
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
        
        # load all data files 
        self.all_files = data_utils.get_data_files(self.root, 'full_data.txt')
        self.num_sample = self.all_files.__len__()
        self.reset() # shuffle frames 
        
        #TODO: split data into test/training set 
        self.split_dir = os.path.join(root, split + '.txt')
        self.sample_list = data_utils.get_data_files(self.split_dir)

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

        # initialize ground truth database 
        self.gt_database = None

        if not self.random_select:
            self.logger.warning('random select is False')

        # TODO: create batches 

        # Stage 1: Region Proposal Network
        if cfg.RPN.ENABLED:
            if mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                # self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
                self.logger.info('Load testing samples from %s' % self.split_dir)
                self.logger.info('Done: total test samples %d' % len(self.sample_list))
        # Stage 2: box proposal refinement subnetwork 
        elif cfg.RCNN.ENABLED:
            pass


    def reset(self):
        ''' reset order of file list to shuffle them'''
        self.file_idxs = np.arange(0, len(self.all_files))
        if self.shuffle:
            np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0

    def generate_gt_database(self):
        #TODO: move to own GTDatabaseGenerator Class 
        self.gt_database = []
        for idx, sample in enumerate(self.sample_list):
            path = os.path.join(self.root, sample)
            data, labels, bboxes = data_utils.load_h5(path, bbox=True)  
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
            

    def create_train_test_split(self): 
        #TODO 
        if self.shuffle:
            self.current_data, self.current_label, _ = shuffle_data(self.current_data,self.current_label)

    
    def get_lidar(self, idx):
        # randomly subsample points 
        #TODO: this only works if there are enough points --> elaborate more on how to select/upsample points
        #      maybe change from random sampling to farthest point sampling 

        pt_idxs = np.arange(0, self.num_points)
        np,random.shuffle(pt_idxs)
        current_points = torch.from_numpy(self.data[index, pt_idxs].copy()).float()
        return current_points

    def get_label(self, sample_id, pt_idx):
        current_labels = torch.from_numpy(self.labels[index, pt_idxs].copy()).long()
        raise NotImplementedError    

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        # return self.data, self.labels
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