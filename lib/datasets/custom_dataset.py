#!/usr/bin/env python
'''
    Prepare data from custom dataset for PointRCNN architecture
'''

import os 
import numpy as np 
import pandas as pd 
import pickle
import torch
from torch.utils.data import Dataset

import lib.utils.custom_data_utils as data_utils
import lib.utils.object3d as object3d
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/custom_data/")


class CustomRCNNDataset(Dataset): 
    def __init__(self, root, num_points, split='train', mode='TRAIN', 
                random_select=True, logger=None, intensity_channel=True, rcnn_training_roi_dir=None, 
                rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None, rcnn_eval_feature_dir=None, 
                gt_database_dir=None, single_test_input=False): # batch_size=10, normalize=False, 
        """
        :param root: directory path to the dataset
        :param split: 'train' or 'test'
        :param num_points: number of points to process for each pointcloud (needs to be the same)
        :param normalize: whether include the normalized coords in features (default: False)
        :param intensity_channel: whether to include the intensity value to xyz coordinates (default: True)
        :param single_test_input: try the network with just one single input frame (default: False)
        """
        self.root = os.path.join(root, 'custom_data')
        self.split = split 
        self.logger = logger
        self.num_points = num_points # TODO: define number of points to train with per frame
        # self.batch_size = batch_size
        # self.normalize = normalize
        self.intensity_channel = intensity_channel 
        # self.shuffle = shuffle
        self.classes = ('Pedestrian')
        self.num_class = self.classes.__len__()
        
        # load all data files 
        self.all_files = data_utils.get_data_files(os.path.join(self.root, 'full_data.txt'))

        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = []
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
        if single_test_input: # this is for trying network architecture with single input frame 
            if self.mode == 'TRAIN': 
                self.split_dir = os.path.join(self.root, 'train_trial.txt')
            elif self.mode == 'EVAL': 
                self.split_dir = os.path.join(self.root, 'test_trial.txt')
        else:
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
        """ 
        Return bbox annotations per frame, defined as (N,7), i.e. (N x [x, y, z, h, w, l, ry])
        
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
        bbox_obj_list = [object3d.Object3d(box, gt=True) for box in bbox_list]
        return bbox_list

    def __len__(self):
        # TODO: validate this setting also for RCNN 
        return len(self.sample_id_list)

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
    
    # ------------- RPN Functions --------------------

    def get_rpn_sample(self, index):
        """ Prepare input for region proposal network. 

        Args:
            index (int): The index of the point cloud instance, i.e. the corresp. frame. 
        """
        sample_id = int(self.sample_id_list[index])

        pts_lidar = self.get_lidar(sample_id)
        labels = self.get_label(sample_id)
        if self.intensity_channel:
            pts_intensity = pts_lidar[:, 3]
            # normalize intensity values by min, max possible values (0,255) & translate intensity to [-0.5, 0.5]
            pts_intensity_norm = ((pts_intensity - 0) / (255 - 0)).reshape(-1,1) - 0.5
        
        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}

        # generate inputs
        pts_coor = pts_lidar[:,:3]
        dist = np.linalg.norm(pts_lidar[:, 0:3], axis=1)
        # dist = np.sqrt(np.sum(pts_coor**2, axis=1,keepdims=True))
        # print(dist)
        if self.mode == "TRAIN" or self.random_select: 
            if self.num_points < len(pts_lidar): # downsample points 
                # flag for near points 
                dist_flag = dist < 8.0 # initial value for cars was 40 -> choose smaller value for indoor setting 
                far_inds = np.where(dist_flag == 0)[0]
                near_inds = np.where(dist_flag == 1)[0]

                near_inds_choice = np.random.choice(near_inds, self.num_points - len(far_inds), replace=False)
                if self.num_points > len(far_inds): 
                    choice = np.concatenate((near_inds_choice, far_inds), axis=0) if len(far_inds) > 0 else near_inds_choice
                else: 
                    choice = np.arange(0, len(self.num_points), dtype=np.int32)
                    choice = np.random.choice(choice, self.num_points, replace=False)
            else:
                choice = np.arange(0, len(pts_lidar), dtype=np.int32)
                if self.num_points > len(pts_lidar): # upsample points by randomly doubling existent points
                    extra_choice = np.random.choice(choice, self.num_points - len(pts_lidar), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
            
            np.random.shuffle(choice)
            pts_coor = pts_coor[choice,:]
            pts_features = [pts_intensity_norm[choice,:]]
            ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((pts_coor, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = pts_coor
        
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = pts_input
        sample_info['pts_features'] = pts_intensity_norm[choice,:]
        
        # stop here if only testing 
        if self.mode == 'TEST':    
            return sample_info

        # prepare 3d ground truth bound boxes sss
        gt_bbox_list = self.get_bbox_label(index)
        # gt_obj_list = [object3d.Object3d(box_annot, gt=True) for box_annot in gt_bbox_list]
        gt_boxes3d = kitti_utils.objs_to_boxes3d_velodyne(gt_obj_list)

        #TODO: data augmentation

        # generate training labels 
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(pts_coor, gt_boxes3d)
        # rpn_cls_label = (labels[choice,:]).astype(np.float32)
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
            center3d[2] += gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_coor  # Now z is the true center of 3d box 

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

    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
        if cfg.RCNN.USE_SEG_SCORE:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    # ------------- RCNN Functions --------------------
    
    def get_proposal_from_file(self, index): 
        """ 
            If proposals from first stage were saved to txt files, they can be directly loaded. 
        """
        sample_id = int(self.image_idx_list[index])
        proposal_file = os.path.join(self.rcnn_eval_roi_dir, '%06d.txt' % sample_id)
        # get detections from output file of stage 1 
        roi_obj_list = kitti_utils.get_objects_from_label(proposal_file)

        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.get_rpn_features(self.rcnn_eval_feature_dir, sample_id)
        pts_rect, pts_rpn_features, pts_intensity = rpn_xyz, rpn_features, rpn_intensity

        roi_box3d_list, roi_scores = [], []
        for obj in roi_obj_list:
            box3d = np.array([obj.pos[0], obj.pos[1], obj.pos[2], obj.h, obj.w, obj.l, obj.ry], dtype=np.float32)
            roi_box3d_list.append(box3d.reshape(1, 7))
            roi_scores.append(obj.score)

        roi_boxes3d = np.concatenate(roi_box3d_list, axis=0)  # (N, 7)
        roi_scores = np.array(roi_scores, dtype=np.float32)  # (N)

        if cfg.RCNN.ROI_SAMPLE_JIT:
            sample_dict = {'sample_id': sample_id,
                           'rpn_xyz': rpn_xyz,
                           'rpn_features': rpn_features,
                           'seg_mask': seg_mask,
                           'roi_boxes3d': roi_boxes3d,
                           'roi_scores': roi_scores,
                           'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

            if self.mode != 'TEST':
                gt_obj_list = self.get_bbox_label(sample_id)
                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

                roi_corners = kitti_utils.boxes3d_to_corners3d_velodyne(roi_boxes3d)
                gt_corners = kitti_utils.boxes3d_to_corners3d_velodyne(gt_boxes3d)
                iou3d = kitti_utils.get_iou3d_velodyne(roi_corners, gt_corners)
                if gt_boxes3d.shape[0] > 0:
                    gt_iou = iou3d.max(axis=1)
                else:
                    gt_iou = np.zeros(roi_boxes3d.shape[0]).astype(np.float32)

                sample_dict['gt_boxes3d'] = gt_boxes3d
                sample_dict['gt_iou'] = gt_iou
            return sample_dict

        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [pts_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            cur_depth = np.linalg.norm(pts_rect, axis=1, ord=2)
            cur_depth_norm = (cur_depth / 20.0) - 0.5
            pts_extra_input_list.append(cur_depth_norm.reshape(-1, 1))

        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)
        pts_input, pts_features = roipool3d_utils.roipool3d_cpu(roi_boxes3d, pts_rect, pts_rpn_features,
                                                                pts_extra_input, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                sampled_pt_num=cfg.RCNN.NUM_POINTS)

        sample_dict = {'sample_id': sample_id,
                       'pts_input': pts_input,
                       'pts_features': pts_features,
                       'roi_boxes3d': roi_boxes3d,
                       'roi_scores': roi_scores,
                       'roi_size': roi_boxes3d[:, 3:6]}

        if self.mode == 'TEST':
            return sample_dict

        gt_obj_list = self.get_bbox_label(sample_id)
        gt_boxes3d = np.zeros((gt_obj_list.__len__(), 7), dtype=np.float32)

        for k, obj in enumerate(gt_obj_list):
            gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                = obj.pos, obj.h, obj.w, obj.l, obj.ry

        if gt_boxes3d.__len__() == 0:
            gt_iou = np.zeros((roi_boxes3d.shape[0]), dtype=np.float32)
        else:
            roi_corners = kitti_utils.boxes3d_to_corners3d_velodyne(roi_boxes3d)
            gt_corners = kitti_utils.boxes3d_to_corners3d_velodyne(gt_boxes3d)
            iou3d = kitti_utils.get_iou3d_velodyne(roi_corners, gt_corners)
            gt_iou = iou3d.max(axis=1)
        sample_dict['gt_boxes3d'] = gt_boxes3d
        sample_dict['gt_iou'] = gt_iou

        return sample_dict
    
    def get_rcnn_sample_info(self, roi_info):
        sample_id, gt_box3d = roi_info['sample_id'], roi_info['gt_box3d']
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.rpn_feature_list[sample_id]

        # augmentation original roi by adding noise
        roi_box3d = self.aug_roi_by_noise(roi_info)

        # point cloud pooling based on roi_box3d
        pooled_boxes3d = kitti_utils.enlarge_box3d(roi_box3d.reshape(1, 7), cfg.RCNN.POOL_EXTRA_WIDTH)

            # inside/outside test if point inside enlarged bbox 
        boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(rpn_xyz),
                                                                 torch.from_numpy(pooled_boxes3d))
        pt_mask_flag = (boxes_pts_mask_list[0].numpy() == 1)
        cur_pts = rpn_xyz[pt_mask_flag].astype(np.float32)

        # data augmentation
        aug_pts = cur_pts.copy()
        aug_gt_box3d = gt_box3d.copy().astype(np.float32)
        aug_roi_box3d = roi_box3d.copy()

        #TODO: 
        # if cfg.AUG_DATA and self.mode == 'TRAIN':
        #     # calculate alpha by ry
        #     temp_boxes3d = np.concatenate([aug_roi_box3d.reshape(1, 7), aug_gt_box3d.reshape(1, 7)], axis=0)
        #     temp_x, temp_y, temp_rz = temp_boxes3d[:, 0], temp_boxes3d[:, 1], temp_boxes3d[:, 6]
        #     temp_beta = np.arctan2(temp_y, temp_x).astype(np.float64)
        #     temp_alpha = -np.sign(temp_beta) * np.pi / 2 + temp_beta + temp_rz

        #     # data augmentation
        #     aug_pts, aug_boxes3d, aug_method = self.data_augmentation(aug_pts, temp_boxes3d, temp_alpha, mustaug=True, stage=2)
        #     aug_roi_box3d, aug_gt_box3d = aug_boxes3d[0], aug_boxes3d[1]
        #     aug_gt_box3d = aug_gt_box3d.astype(gt_box3d.dtype)

        # Pool input points
        valid_mask = 1  # whether the input is valid

        if aug_pts.shape[0] == 0:
            pts_features = np.zeros((1, 128), dtype=np.float32)
            input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            pts_input = np.zeros((1, input_channel), dtype=np.float32)
            valid_mask = 0
        else:
            pts_features = rpn_features[pt_mask_flag].astype(np.float32)
            pts_intensity = rpn_intensity[pt_mask_flag].astype(np.float32)

            pts_input_list = [aug_pts, pts_intensity.reshape(-1, 1)]
            if cfg.RCNN.USE_INTENSITY:
                pts_input_list = [aug_pts, pts_intensity.reshape(-1, 1)]
            else:
                pts_input_list = [aug_pts]

            if cfg.RCNN.USE_MASK:
                if cfg.RCNN.MASK_TYPE == 'seg':
                    pts_mask = seg_mask[pt_mask_flag].astype(np.float32)
                elif cfg.RCNN.MASK_TYPE == 'roi':
                    pts_mask = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(aug_pts),
                                                                  torch.from_numpy(aug_roi_box3d.reshape(1, 7)))
                    pts_mask = (pts_mask[0].numpy() == 1).astype(np.float32)
                else:
                    raise NotImplementedError

                pts_input_list.append(pts_mask.reshape(-1, 1))

            if cfg.RCNN.USE_DEPTH:
                pts_depth = np.linalg.norm(aug_pts, axis=1, ord=2)
                pts_depth_norm = (pts_depth / 20.0) - 0.5 # scale depth with max distance of 20
                pts_input_list.append(pts_depth_norm.reshape(-1, 1))

            pts_input = np.concatenate(pts_input_list, axis=1)  # (N, C)

        aug_gt_corners = kitti_utils.boxes3d_to_corners3d_velodyne(aug_gt_box3d.reshape(-1, 7))
        aug_roi_corners = kitti_utils.boxes3d_to_corners3d_velodyne(aug_roi_box3d.reshape(-1, 7))
        iou3d = kitti_utils.get_iou3d_velodyne(aug_roi_corners, aug_gt_corners)
        cur_iou = iou3d[0][0]

        # regression valid mask
        reg_valid_mask = 1 if cur_iou >= cfg.RCNN.REG_FG_THRESH and valid_mask == 1 else 0

        # classification label
        cls_label = 1 if cur_iou > cfg.RCNN.CLS_FG_THRESH else 0
        if cfg.RCNN.CLS_BG_THRESH < cur_iou < cfg.RCNN.CLS_FG_THRESH or valid_mask == 0:
            cls_label = -1

        # canonical transform and sampling
        pts_input_ct, gt_box3d_ct = self.canonical_transform(pts_input, aug_roi_box3d, aug_gt_box3d)
        pts_input_ct, pts_features = self.rcnn_input_sample(pts_input_ct, pts_features)

        sample_info = {'sample_id': sample_id,
                       'pts_input': pts_input_ct,
                       'pts_features': pts_features,
                       'cls_label': cls_label,
                       'reg_valid_mask': reg_valid_mask,
                       'gt_boxes3d_ct': gt_box3d_ct,
                       'roi_boxes3d': aug_roi_box3d,
                       'roi_size': aug_roi_box3d[3:6],
                       'gt_boxes3d': aug_gt_box3d}

        return sample_info
    
    def get_rcnn_training_sample_batch(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.get_bbox_label(sample_id)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        # calculate original iou
        iou3d = kitti_utils.get_iou3d_velodyne(kitti_utils.boxes3d_to_corners3d(roi_boxes3d),
                                      kitti_utils.boxes3d_to_corners3d(gt_boxes3d))
        max_overlaps, gt_assignment = iou3d.max(axis=1), iou3d.argmax(axis=1)
        max_iou_of_gt, roi_assignment = iou3d.max(axis=0), iou3d.argmax(axis=0)
        roi_assignment = roi_assignment[max_iou_of_gt > 0].reshape(-1)

        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        fg_inds = np.nonzero(max_overlaps >= fg_thresh)[0]
        fg_inds = np.concatenate((fg_inds, roi_assignment), axis=0)  # consider the roi which has max_overlaps with gt as fg

        easy_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO))[0]
        hard_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                  (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO))[0]

        fg_num_rois = fg_inds.size
        bg_num_rois = hard_bg_inds.size + easy_bg_inds.size

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = np.random.permutation(fg_num_rois)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE  - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE ) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
            fg_rois_per_this_image = 0
        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError

        # augment the rois by noise
        roi_list, roi_iou_list, roi_gt_list = [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois_src = roi_boxes3d[fg_inds].copy()
            gt_of_fg_rois = gt_boxes3d[gt_assignment[fg_inds]]
            fg_rois, fg_iou3d = self.aug_roi_by_noise_batch(fg_rois_src, gt_of_fg_rois, aug_times=10)
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)

        if bg_rois_per_this_image > 0:
            bg_rois_src = roi_boxes3d[bg_inds].copy()
            gt_of_bg_rois = gt_boxes3d[gt_assignment[bg_inds]]
            bg_rois, bg_iou3d = self.aug_roi_by_noise_batch(bg_rois_src, gt_of_bg_rois, aug_times=1)
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)

        rois = np.concatenate(roi_list, axis=0)
        iou_of_rois = np.concatenate(roi_iou_list, axis=0)
        gt_of_rois = np.concatenate(roi_gt_list, axis=0)

        # collect extra features for point cloud pooling
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [rpn_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = (np.linalg.norm(rpn_xyz, ord=2, axis=1) / 70.0) - 0.5
            pts_extra_input_list.append(pts_depth.reshape(-1, 1))
        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)

        pts_input, pts_features, pts_empty_flag = roipool3d_utils.roipool3d_cpu(rois, rpn_xyz, rpn_features,
                                                                                pts_extra_input,
                                                                                cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                                sampled_pt_num=cfg.RCNN.NUM_POINTS,
                                                                                canonical_transform=False)

        # data augmentation
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            for k in range(rois.__len__()):
                aug_pts = pts_input[k, :, 0:3].copy()
                aug_gt_box3d = gt_of_rois[k].copy()
                aug_roi_box3d = rois[k].copy()

                # calculate alpha by ry
                temp_boxes3d = np.concatenate([aug_roi_box3d.reshape(1, 7), aug_gt_box3d.reshape(1, 7)], axis=0)
                temp_x, temp_z, temp_ry = temp_boxes3d[:, 0], temp_boxes3d[:, 2], temp_boxes3d[:, 6]
                temp_beta = np.arctan2(temp_z, temp_x).astype(np.float64)
                temp_alpha = -np.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry

                # data augmentation
                aug_pts, aug_boxes3d, aug_method = self.data_augmentation(aug_pts, temp_boxes3d, temp_alpha,
                                                                          mustaug=True, stage=2)

                # assign to original data
                pts_input[k, :, 0:3] = aug_pts
                rois[k] = aug_boxes3d[0]
                gt_of_rois[k] = aug_boxes3d[1]

        valid_mask = (pts_empty_flag == 0).astype(np.int32)

        # regression valid mask
        reg_valid_mask = (iou_of_rois > cfg.RCNN.REG_FG_THRESH).astype(np.int32) & valid_mask

        # classification label
        cls_label = (iou_of_rois > cfg.RCNN.CLS_FG_THRESH).astype(np.int32)
        invalid_mask = (iou_of_rois > cfg.RCNN.CLS_BG_THRESH) & (iou_of_rois < cfg.RCNN.CLS_FG_THRESH)
        cls_label[invalid_mask] = -1
        cls_label[valid_mask == 0] = -1

        # canonical transform and sampling
        pts_input_ct, gt_boxes3d_ct = self.canonical_transform_batch(pts_input, rois, gt_of_rois)

        pts_features = np.concatenate((pts_input_ct[:,:,3:],pts_features), axis=2)
        pts_input_ct = pts_input_ct[:,:,0:3]

        sample_info = {'sample_id': sample_id,
                       'pts_input': pts_input_ct,
                       'pts_features': pts_features,
                       'cls_label': cls_label,
                       'reg_valid_mask': reg_valid_mask,
                       'gt_boxes3d_ct': gt_boxes3d_ct,
                       'roi_boxes3d': rois,
                       'roi_size': rois[:, 3:6],
                       'gt_boxes3d': gt_of_rois}

        return sample_info

    @staticmethod
    def rcnn_input_sample(pts_input, pts_features):
        choice = np.random.choice(pts_input.shape[0], cfg.RCNN.NUM_POINTS, replace=True)

        if pts_input.shape[0] < cfg.RCNN.NUM_POINTS:
            choice[:pts_input.shape[0]] = np.arange(pts_input.shape[0])
            np.random.shuffle(choice)
        pts_input = pts_input[choice]
        pts_features = pts_features[choice]

        return pts_input, pts_features
    
    def aug_roi_by_noise(self, roi_info):
        """
        add noise to original roi to get aug_box3d
        :param roi_info:
        :return:
        """
        roi_box3d, gt_box3d = roi_info['roi_box3d'], roi_info['gt_box3d']
        original_iou = roi_info['iou3d']
        temp_iou = cnt = 0
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        gt_corners = kitti_utils.boxes3d_to_corners3d_velodyne(gt_box3d.reshape(-1, 7))
        aug_box3d = roi_box3d
        while temp_iou < pos_thresh and cnt < 10:
            if roi_info['type'] == 'gt':
                aug_box3d = self.random_aug_box3d(roi_box3d)  # GT, must random
            else:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
            aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(-1, 7))
            iou3d = kitti_utils.get_iou3d_velodyne(aug_corners, gt_corners)
            temp_iou = iou3d[0][0]
            cnt += 1
            if original_iou < pos_thresh:  # original bg, break
                break
        return aug_box3d

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, rz]
        random shift, scale, orientation
        """
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            pos_shift = (np.random.rand(3) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (np.random.rand(3) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (np.random.rand(1) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]

            aug_box3d = np.concatenate([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale,
                                        box3d[6:7] + angle_rot])
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]
            idx = np.random.randint(len(range_config))

            pos_shift = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((np.random.rand(1) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = np.concatenate([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot])
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.3)
            z_shift = np.random.normal(loc=0, scale=0.2)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.5)
            l_shift = np.random.normal(loc=0, scale=0.15)
            rz_shift = ((np.random.rand() - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                  box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + rz_shift])
            return aug_box3d
        else:
            raise NotImplementedError

    @staticmethod
    def canonical_transform(pts_input, roi_box3d, gt_box3d):
        roi_rz = roi_box3d[6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_box3d[0:3]
        # shift to center
        pts_input[:, [0, 1, 2]] = pts_input[:, [0, 1, 2]] - roi_center
        gt_box3d_ct = np.copy(gt_box3d)
        gt_box3d_ct[0:3] = gt_box3d_ct[0:3] - roi_center
        # rotate to the direction of head
        gt_box3d_ct = kitti_utils.rotate_pc_along_z(gt_box3d_ct.reshape(1, 7), roi_rz).reshape(7)
        gt_box3d_ct[6] = gt_box3d_ct[6] - roi_rz
        pts_input = kitti_utils.rotate_pc_along_z(pts_input, roi_ry)

        return pts_input, gt_box3d_ct

    @staticmethod
    def canonical_transform_batch(pts_input, roi_boxes3d, gt_boxes3d):
        """
        :param pts_input: (N, npoints, 3 + C)
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        roi_rz = roi_boxes3d[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_boxes3d[:, 0:3]
        # shift to center
        pts_input[:, :, [0, 1, 2]] = pts_input[:, :, [0, 1, 2]] - roi_center.reshape(-1, 1, 3)
        gt_boxes3d_ct = np.copy(gt_boxes3d)
        gt_boxes3d_ct[:, 0:3] = gt_boxes3d_ct[:, 0:3] - roi_center
        # rotate to the direction of head
        gt_boxes3d_ct = kitti_utils.rotate_pc_along_z(torch.from_numpy(gt_boxes3d_ct.reshape(-1, 1, 7)).float(),
                                                      torch.from_numpy(roi_rz).float()).numpy().reshape(-1, 7)
        gt_boxes3d_ct[:, 6] = gt_boxes3d_ct[:, 6] - roi_rz
        pts_input = kitti_utils.rotate_pc_along_z(torch.from_numpy(pts_input).float(), 
                                                 torch.from_numpy(roi_rz).float()).numpy()

        return pts_input, gt_boxes3d_ct
        
    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, gt_alpha, sample_id=None, mustaug=False, stage=1):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []

        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            # rotate around z-axis 
            rot_range = [-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE]
            # xyz change, hwl unchange
            aug_gt_boxes3d, aug_pts_rect = kitti_utils.global_rotation(aug_gt_boxes3d, aug_pts_rect, rot_range)
            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect *= scale
            aug_gt_boxes3d[:, 0:6] *= scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # random horizontal flip along y axis 
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            aug_gt_boxes3d[:, 6] = -(aug_gt_boxes3d[:, 6] + np.pi)   
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]

            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7] = -gt_boxes[:, 7]
            aug_method.append('flip')

        return aug_pts_rect, aug_gt_boxes3d, aug_method
        
    def get_rcnn_sample_jit(self, index):
        raise NotImplementedError

if __name__ == '__main__':
    pass
