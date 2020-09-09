'''
    Taken from the following repository: 
    https://github.com/open-mmlab/OpenPCDet/blob/master/tools/visual_utils/visualize_utils.py
'''
import _init_path
import argparse
import vtk
from vtk.util import numpy_support
import mayavi.mlab as mlab
import numpy as np
import torch
import os

import lib.utils.custom_data_utils as data_utils
from lib.utils.kitti_utils import boxes3d_to_corners3d_velodyne

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--id', type=int, default=0, required=True, help='Specify the sample id to be evaluated/visualized.')
parser.add_argument('--train_run_id', type=int, default=1, help='Specify the id of the trained network to be evaluated/visualized.')
parser.add_argument('--epoch_no', type=int, default=5, help='Specify the epoch number.')

args = parser.parse_args()


def generate_corners3d(bbox):
    """
    Generate corners3d representation (oriented bounding box representation) for this object
    :h,w,l bounding box dimensions 
    :rz rotation angle around z-axis in velodyn coord. (-pi, pi)
        7 -------- 6
       /|         /|
      4 -------- 5 .
      | |        | |
      . 3 -------- 2
      |/         |/
      0 -------- 1
    :return corners_3d: (8, 3) corners of oriented box3d in Velodyne coord.
    """
    pos,h,w,l,rz = bbox[0:3], bbox[3], bbox[4], bbox[5], bbox[6]
    # careful: width, length and height have been differently defined than in KITTI
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]        
    y_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

    # rotation now defined in Velodyne coords. -> around z-axis => yaw rot. 
    R = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [np.sin(rz), np.cos(rz), 0],
                  [0, 0, 1]])
    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    # transpose and rotate around orientation angle 
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + pos
    return np.reshape(corners3d, (1, 8,3))


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(1000, 1000), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='spectral', scale_factor=10, scale_mode='vector', figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=10, figure=fig)
    if draw_origin:
        #draw origin
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    
    # draw fov (todo: update to real sensor spec.)
    # fov=np.array([  # 45 degree
    #     [20., 20., 0.,0.],
    #     [20.,-20., 0.,0.],
    # ],dtype=np.float64)

    # mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    # mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    axes=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 2, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 6
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, foreground_pts=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points, show_intensity=True)
    fig = draw_multi_grid_range(fig, bv_range=(0, -20, 40, 20))
    if gt_boxes is not None:
        corners3d = boxes3d_to_corners3d_velodyne(gt_boxes)
        # box = gt_boxes[0,:]
        # corners3d = generate_corners3d(box)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None:
        # ref_corners3d = boxes3d_to_corners3d_velodyne(ref_boxes)
        ref_box = ref_boxes[0,:]
        ref_corners3d = generate_corners3d(ref_box)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    # plot foreground segmentation results
    if foreground_pts is not None:
        fig = draw_sphere_pts(foreground_pts, fig=fig)
    mlab.view(azimuth=-180, elevation=54.0, distance=62.0, roll=90.0, figure=fig, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])
    return fig


def readIntoNumpy(fileName):
    tmp = []
    scores = []
    with open(fileName) as f:
        lines = []
        for line in f :
            splitLine = line.rstrip().split()
            res = splitLine[3:10]   # [x y z h w l ry]
            score = splitLine[10]
            tmp.append(res)
            scores.append(score)
    bboxes3d = np.array(tmp,  dtype=np.float32)
    return bboxes3d, scores


if __name__ == "__main__":
    DATA_PATH = os.path.join('../../', 'data/custom_data/')
    OUTPUT_PATH = os.path.join('../../', 'output/')
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