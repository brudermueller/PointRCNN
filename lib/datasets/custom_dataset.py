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
    def __init__(self, root, num_points, split='train', mode='TRAIN', 
                random_select=True, logger=None, intensity_channel=True, rcnn_training_roi_dir=None, 
                rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None, rcnn_eval_feature_dir=None, 
                gt_database_dir=None): # batch_size=10, normalize=False, 
        """
        :param root: directory path to the dataset
        :param split: 'train' or 'test'
        :param num_points: number of points to process for each pointcloud (needs to be the same)
        :param normalize: whether include the normalized coords in features (default: False)
        :param intensity_channel: whether to include the intensity value to xyz coordinates (default: True)
        """
        self.root = os.path.join(root, 'custom_data')
        self.split = split 
        self.logger = logger
        self.num_points = num_points # TODO: define number of points to train with per frame
        # self.batch_size = batch_size
        # self.normalize = normalize
        self.intensity_channel = intensity_channel 
        # self.shuffle = shuffle
        self.classes = ('Background', 'Pedestrian')
        self.num_class = self.classes.__len__()
        
        # load all data files 
        self.all_files = data_utils.get_data_files(os.path.join(self.root, 'full_data.txt'))

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
        self.split_dir = os.path.join(self.root, split + '.txt')
        self.logger.info('Load samples from %s' % self.split_dir)
        self.current_samples = data_utils.get_data_files(self.split_dir)
        
        # Create Mapping from sample frames to frame ids 
        self.sample_id_list = [idx for idx in range(0, self.current_samples.__len__())]
        self.num_sample = self.sample_id_list.__len__()

        # self.num_sample = self.all_files.__len__()
        self.logger.info('Done: total {}-samples {}'.format(self.split, len(self.current_samples)))

    def get_lidar(self, index):        
        """ Returns lidar point data loaded from h5 file in form of (N,4).

        Args:
            frame (string): frame id 
        """
        frame = self.current_samples[index]
        # print('++++++++ Frame {} +++++++++'.format(frame))
        lidar_file = os.path.join(self.root, frame)
        assert os.path.exists(lidar_file)
        pts, _ = data_utils.load_h5(lidar_file)
        return pts

    def get_label(self, index):
        """ Returns point labels for each point in lidar data loaded from h5 file in form of (N,1).

        Args:
            frame (string): frame id 
        """
        frame = self.current_samples[index]
        lidar_file = os.path.join(self.root, frame)
        assert os.path.exists(lidar_file)
        _, labels = data_utils.load_h5(lidar_file)
        return np.reshape(labels, (-1,1))

    def get_bbox_label(self, index):
        """ Return bbox annotations per frame, defined as (N,7), i.e. (N x [x, y, z, h, w, l, ry])

        Args:
            frame (string): frame id 
        """
        frame = self.current_samples[index]
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
        sample_id = int(self.sample_id_list[index])

        pts_lidar = self.get_lidar(sample_id)
        labels = self.get_label(sample_id)
        if self.intensity_channel:
            pts_intensity = pts_lidar[:, 3].reshape(-1,1)
        
        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}

        # generate inputs
        pts_coor = pts_lidar[:,:3]
        dist = np.sqrt(np.sum(pts_coor**2, axis=1,keepdims=True))
        # print(dist)
        if self.mode == "TRAIN" or self.random_select: 
            if self.num_points < len(pts_lidar): # downsample points 
                dist_flag = dist < 8.0 # initial value for cars was 40 -> choose smaller value for indoor setting 
                far_inds = np.where(dist_flag == 0)[0]
                near_inds = np.where(dist_flag == 1)[0]

                near_inds_choice = np.random.choice(near_inds, self.num_points - len(far_inds), replace=False)
                choice = np.concatenate((near_inds_choice, far_inds), axis=0) if len(far_inds) > 0 else near_inds_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_lidar), dtype=np.int32)
                if self.num_points > len(pts_lidar): # upsample points by randomly doubling existent points
                    extra_choice = np.random.choice(choice, self.num_points - len(pts_lidar), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)
            
            pts_coor = pts_coor[choice,:]
            pts_features = pts_intensity[choice,:]
  
        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((pts_coor, pts_features), axis=1)  # (N, C)
        else:
            pts_input = pts_coor
        
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = pts_input
        sample_info['pts_features'] = pts_intensity[choice,:]
        
        # stop here if only testing 
        if self.mode == 'TEST':    
            return sample_info

        # prepare 3d ground truth bound boxes sss
        gt_bbox_list = self.get_bbox_label(index)
        gt_obj_list = [object3d.CustomObject3d(box_annot) for box_annot in gt_bbox_list]
        gt_boxes3d = kitti_utils.objs_to_boxes3d_velodyne(gt_obj_list)

        #TODO: data augmentation

        # generate training labels 
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(pts_coor, gt_boxes3d)
        # rpn_cls_label = (labels[choice,:]).astype(np.float32)
        rpn_reg_label = rpn_reg_label 
        sample_info['rpn_cls_label'] =  rpn_cls_label # 0:background, 1: pedestrian
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = gt_boxes3d
        return sample_info 

    @staticmethod
    def generate_rpn_training_labels(pts_coor, gt_boxes3d):
        # bottom up 3d bbox regression from foreground points during training
        cls_label = np.zeros((pts_coor.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_coor.shape[0], 7), dtype=np.float32)  # dx, dy, dz, rz, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d_velodyne(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d_velodyne(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_coor, box_corners)
            fg_pts_coor = pts_coor[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_coor, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_coor  # Now y is the true center of 3d box 

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # rz

        return cls_label, reg_label

    def collate_batch(self, batch): 
        """ Merge list of samples to create mini-batch

        Args:
            batch ([type]): [description]
        """
        # testing 
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
    
    def get_rcnn_sample_jit(self, index):
        raise NotImplementedError
    
    def get_rcnn_training_sample_batch(self, index):
        raise NotImplementedError
    
    def get_proposal_from_file(self, index):
        raise NotImplementedError



if __name__ == '__main__':
    pass
