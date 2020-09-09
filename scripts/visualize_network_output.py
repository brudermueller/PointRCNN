import _init_path
import argparse
import os
import numpy as np
import mayavi.mlab as mlab


import lib.utils.custom_data_utils as data_utils
from tools.plot_utils.visualize_utils import readIntoNumpy, draw_scenes


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--id', type=int, default=0, required=True, help='Specify the sample id to be evaluated/visualized.')
parser.add_argument('--train_run_id', type=int, default=1, help='Specify the id of the trained network to be evaluated/visualized.')
parser.add_argument('--epoch_no', type=int, default=5, help='Specify the epoch number.')

args = parser.parse_args()



if __name__ == "__main__":
    DATA_PATH = os.path.join('../', 'data/custom_data/')
    OUTPUT_PATH = os.path.join('../', 'output/')
    all_val_files = data_utils.get_data_files(os.path.join(DATA_PATH, 'val.txt'))
    # path of lidar frame
    idx = args.id
    lidar_file = os.path.join(DATA_PATH, all_val_files[idx])

    assert os.path.exists(lidar_file)
    pts, _, bboxes = data_utils.load_h5(lidar_file, bbox=True)

    # path of output from model
    if idx >= 1000: digit = str(idx)
    elif idx >= 100: digit = '0' + str(idx)
    elif idx >=10: digit = '00' + str(idx)
    else: digit = '000' + str(idx)
    bboxes3d_path = os.path.join(OUTPUT_PATH, "rpn/pedestrian{}/eval/epoch_{}/val/detections/data/00{}.txt".format(args.train_run_id, args.epoch_no, digit))
    bboxes3d, scores = readIntoNumpy(bboxes3d_path)
    best_box_idx = np.argmax(scores)
    gt_boxes = np.reshape(bboxes, (-1, 7))

    # load foreground segmentation results 
    seg_pts_file = os.path.join(OUTPUT_PATH, "rpn/pedestrian{}/eval/epoch_{}/val/seg_result/00{}.h5".format(args.train_run_id, args.epoch_no, digit))
    seg_pts = data_utils.load_h5_basic(seg_pts_file)
    mask = seg_pts[:,4] > 0 
    foreground = seg_pts[mask, :][:, 0:3]


    # fig = draw_scenes(pts, gt_boxes=np.reshape(bboxes3d[best_box_idx,:], (-1,7)), ref_boxes=gt_boxes, foreground_pts=foreground)
    fig = draw_scenes(pts, gt_boxes=np.reshape(bboxes3d, (-1,7)), ref_boxes=gt_boxes, foreground_pts=foreground)
    # fig = draw_scenes(pts, gt_boxes=np.reshape(bboxes3d[best_box_idx,:], (-1,7)), ref_boxes=gt_boxes)

    mlab.show()