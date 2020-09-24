import torch
import numpy as np


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, [0, 2]].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).squeeze(dim=1)
    return pc


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    # print(points_rot.size())
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                       get_xy_fine=True, get_z_by_bin=False, loc_z_scope=0.5, loc_z_bin_size=0.25, get_rz_fine=False):
    """
    This function is used in both network stages (RPN and RCNN) in order to recover the 3D bounding boxes from regression. 
    :param roi_box3d: (N, 7) for RCNN, (N,3) for RPN # backbone_xyz (coordinates of foreground points)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :param get_z_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_rz_fine:
    :return:
    """
    # print('=================== Decode bbox input =================')
    # print('roibox3d input: {}'.format(roi_box3d.size()))
    # print('pred_reg: {}'.format(pred_reg.size()))
    anchor_size = anchor_size.to(roi_box3d.get_device()) #transform anchor box into cuda tensor
    
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2 # number of bins for x, z axes around point
    loc_z_bin_num = int(loc_z_scope / loc_z_bin_size) * 2 

    # recover xz localization
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    y_bin_l, y_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = y_bin_r

    x_bin = torch.argmax(pred_reg[:, x_bin_l: x_bin_r], dim=1)
    y_bin = torch.argmax(pred_reg[:, y_bin_l: y_bin_r], dim=1)

    # pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    # pos_y = y_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope

    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 + loc_scope
    pos_y = y_bin.float() * loc_bin_size + loc_bin_size / 2 + loc_scope

    if get_xy_fine: # increase number of bin boxes for better resolution 
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3 
        y_res_l, y_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = y_res_r

        x_res_norm = torch.gather(pred_reg[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        y_res_norm = torch.gather(pred_reg[:, y_res_l: y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
        x_res = x_res_norm * loc_bin_size
        y_res = y_res_norm * loc_bin_size

        pos_x += x_res
        pos_y += y_res

    # recover z localization
    if get_z_by_bin:
        z_bin_l, z_bin_r = start_offset, start_offset + loc_z_bin_num
        z_res_l, z_res_r = z_bin_r, z_bin_r + loc_z_bin_num
        start_offset = z_res_r

        z_bin = torch.argmax(pred_reg[:, z_bin_l: z_bin_r], dim=1)
        z_res_norm = torch.gather(pred_reg[:, z_res_l: z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
        z_res = z_res_norm * loc_z_bin_size
        pos_z = z_bin.float() * loc_z_bin_size + loc_z_bin_size / 2 - loc_z_scope + z_res
        pos_z = pos_z + roi_box3d[:, 2]
    else:
        z_offset_l, z_offset_r = start_offset, start_offset + 1
        start_offset = z_offset_r
        pos_z = roi_box3d[:, 2] + pred_reg[:, z_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
    if get_rz_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
    else:
        angle_per_class = (2 * np.pi) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)

        # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2 * np.pi

    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l: size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d

    if roi_box3d.shape[1] == 7: # for RCNN stage 2 
        roi_ry = roi_box3d[:, 6]
        # print(roi_ry.size())
        # ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, - roi_ry)
        
        shift_ret_box3d = shift_ret_box3d.unsqueeze(dim=1)
        shape = list(shift_ret_box3d.size())
        # print(shape)
        ret_box3d = rotate_points_along_z(shift_ret_box3d, -roi_ry).squeeze(dim=1)
        # print(ret_box3d.size())
        ret_box3d[:, 6] += roi_ry

    ret_box3d[:, [0, 1]] += roi_center[:, [0, 1]]

    return ret_box3d
