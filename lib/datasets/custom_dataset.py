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
import lib.utils.custom_object3d as object3d
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/custom_data/")


class CustomRCNNDataset(Dataset): 
    def __init__(self, root, num_points, split='train', mode='TRAIN', batch_size=10, normalize=False, 
                random_select=True, logger=None, intensity_channel=True, rcnn_training_roi_dir=None, 
                rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                rcnn_eval_feature_dir=None): 
        """
        :param root: directory path to the dataset
        :param split: 'train' or 'test'
        :param num_points: number of points to process for each pointcloud (needs to be the same)
        :param normalize: whether include the normalized coords in features (default: False)
        :param intensity_channel: whether to include the intensity value to xyz coordinates (default: True)
        """
        self.root = root
        self.split = split 
        self.logger = logger
        self.num_points = num_points # TODO: define number of points to train with per frame
        self.batch_size = batch_size
        self.normalize = normalize
        self.intensity_channel = intensity_channel 
        self.shuffle = shuffle
        self.classes = ('Background', 'Pedestrian')
        self.num_class = self.classes.__len__()
        
        # load all data files 
        self.all_files = data_utils.get_data_files(self.root, 'full_data.txt')

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

        self.random_select = random_select
        if not self.random_select:
            self.logger.warning('random select is False')

        # TODO: create batches 

        # Stage 1: Region Proposal Network (RPN)
        # Stage 2: box proposal refinement subnetwork (RCNN)

        if cfg.RPN.ENABLED:
            # initialize ground truth database (needed for data augmentation)
            if gt_database_dir is not None: 
                logger.info('Loading gt_database(%d) from %s' % (len(self.gt_database), gt_database_dir))
                self.gt_database = pickle.load(open(gt_database_dir, 'rb'))
            
        # load samples to work with (depending on train/test/val mode)
        self.logger.info('Load samples from %s' % self.split_dir)
        self.split_dir = os.path.join(root, split + '.txt')
        self.current_samples = data_utils.get_data_files(self.split_dir)
        # self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        self.num_sample = self.all_files.__len__()
        self.logger.info('Done: total {}-samples {}'.format(self.split, len(self.current_samples)))

    def get_lidar(self, frame):        
        """ Returns lidar point data loaded from h5 file in form of (N,4).

        Args:
            frame (string): frame name/id 
        """
        lidar_file = os.path.join(self.root, frame)
        assert os.path.exists(lidar_file)
        pts, _ = data_utils.load_h5(lidar_file)
        return pts

    def get_label(self, frame):
        """ Returns point labels for each point in lidar data loaded from h5 file in form of (N,1).

        Args:
            frame (string): frame name/id 
        """
        lidar_file = os.path.join(self.root, frame)
        assert os.path.exists(lidar_file)
        _, labels = data_utils.load_h5(lidar_file)
        return np.reshape(labels, (-1,1))

    def get_bbox_label(self, frame):
        """ Return bbox annotations per frame, defined as (N,7), i.e. (N x [x, y, z, h, w, l, ry])

        Args:
            frame (string): frame name/id 
        """
        lidar_file = os.path.join(self.root, frame)
        assert os.path.exists(lidar_file)
        # point labels not used here, bboxes instead 
        _, _, bbox = data_utils.load_h5(lidar_file, bbox=True)
        # transform single bbox annotation in list for compability reasons (dataset can be extended with >1 bboxes per frame)
        bbox_list = np.reshape(bbox, (1,-1)) 
        return bbox_list

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.current_samples)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.current_samples)
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
        """ Prepare input for region proposal network. 

        Args:
            index (int): The index of the point cloud instance, i.e. the corresp. frame. 
        """
        pts_lidar = self.get_lidar(index)
        labels = self.get_label(index)
        if self.intensity_channel:
            pts_intensity = pts_lidar[:, 3].reshape(-1,1)
        
        sample_info = {'sample_id': index, 'random_select': self.random_select}

        # generate inputs
        pts_coor = pts_lidar[:,:3]
        dist = np.sqrt(np.sum(pts_coor**2, axis=1,keepdims=True))
        # print(dist)
        if self.num_points < len(pts_lidar): # downsample points 
            dist_flag = dist < 8.0 # initial value for cars was 40 -> choose smaller value for indoor setting 
            far_inds = np.where(dist_flag == 0)[0]
            near_inds = np.where(dist_flag == 1)[0]

            near_inds_choice = np.random.choice(near_inds, self.n_sample_points - len(far_inds), replace=False)
            choice = np.concatenate((near_inds_choice, far_inds), axis=0) if len(far_inds) > 0 else near_inds_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pts_lidar), dtype=np.int32)
            if self.num_points > len(pts_lidar): # upsample points by randomly doubling existent points
                extra_choice = np.random.choice(choice, self.n_sample_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        
        pts_coor = pts_coor[choice,:]
        pts_features = pts_intensity[choice,:]
        
        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((pts_coor, pts_features), axis=1)  # (N, C)
        else:
            pts_input = pts_coor
        
        sample_info['pts_input'] = pts_coor[choice,:]
        sample_info['pts_features'] = pts_intensity[choice,:]
        
        # stop here if only testing 
        if self.mode == 'TEST':    
            return sample_info

        # prepare 3d ground truth bound boxes 
        gt_bbox_list = self.get_bbox_label(index)
        gt_boxes3d = [object3d.Object3d(box_annot) for box_annot in gt_bbox_list]

        #TODO: data augmentation
        
        sample_info['rpn_cls_label'] = (labels[choice,:]).astype(np.float32)  # 0:background, 1: pedestrian
        sample_info['gt_boxes3d'] = gt_boxes3d
        return sample_info 



if __name__ == '__main__':
    pass
