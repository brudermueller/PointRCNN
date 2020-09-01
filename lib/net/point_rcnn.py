import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN', logger=None):
        super().__init__()
        self.logger = logger
        self.mode = mode

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            # print('RPN ENABLED')
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data) # dict with keys: 'rpn_cls', 'rpn_reg', 'backbone_xyz', 'backbone_features'
                output.update(rpn_output)

                # Testing of proposal layer: 
                # if self.mode == 'TEST':
                #     with torch.no_grad():
                #         rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                #         backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                #         rpn_scores_raw = rpn_cls[:, :, 0]
                #         # convert into probability values from 0 to 1 
                #         rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                #         self.logger.debug('rpn_scores_norm: max {}, min {}'.format(torch.max(rpn_scores_norm), torch.min(rpn_scores_norm)))
                #         seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float() # predicted segmentation mask from stage 1 {0,1}

                #         # distance from sensor 
                #         pts_depth = torch.norm(backbone_xyz, p=2, dim=2) # p=order of norm, dim=if it is an int, vector norm will be calculated

                #         # proposal layer
                #         # rois = 3d proposal boxes 
                #         rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                #         self.logger.info('==> Proposals form RPN layer: {}'.format(rois.size()))
                        
            # rcnn inference
            if cfg.RCNN.ENABLED:
                self.logger('==> RCNN ENABLED ')
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    # convert into probability values from 0 to 1 
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float() # predicted segmentation mask from stage 1 {0,1}
                    # distance from sensor 
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2) # p=order of norm, dim=if it is an int, vector norm will be calculated

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois # bbox proposals from first stage 
                    output['roi_scores_raw'] = roi_scores_raw # 
                    output['seg_result'] = seg_mask # foreground points mask 

                rcnn_input_info = {'rpn_xyz': backbone_xyz,# point coordinates 
                                   'rpn_features': backbone_features.permute((0, 2, 1)), # C-dim point feature rep. (PointNet)
                                   'seg_mask': seg_mask, # segmentation mask 
                                   'roi_boxes3d': rois, # regions of interest - bbox predictions
                                   'pts_depth': pts_depth} 
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)
                if self.logger: 
                    self.logger.debug('==> RCNN Output: {}'.format(output))

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output
