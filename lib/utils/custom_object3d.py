import numpy as np 


class CustomObject3d(object):
    def __init__(self, bbox_array): 
        #bbox: (array of length 7) [x, y, z, h, w, l, ry]
        self.centroid = bbox[0:3]
        self.h, self.w, self.l = float(bbox[3]), float(bbox[4]), float(bbox[5])
        self.ry =  bbox[6]  # orientation angle 
        # According to KITTI definition
        self.cls_type = 'Pedestrian'
        self.cls_id = 2

    def generate_corners3d(self):
        """
        Generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        # careful: width, length and height have been differently defined than in KITTI
        y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def calc_3d_box(self): 
        """ 
        Generate 3d bounding box with 8 vertices saved as np.array (of shape (8,3)) for the 
        3d box in following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1

        Returns:
            bbox np.ndarray
        """
        x, y, z, h, w, l = self.x, self.y, self.z, self.h, self.w, self.l
        box8 = np.array(
            [
                [
                    x + w / 2,
                    x + w / 2,
                    x - w / 2,
                    x - w / 2,
                    x + w / 2,
                    x + w / 2,
                    x - w / 2,
                    x - w / 2,
                ],
                [
                    y - l / 2,
                    y + l / 2,
                    y + l / 2,
                    y - l / 2,
                    y - l / 2,
                    y + l / 2,
                    y + l / 2,
                    y - l / 2,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    z + h / 2,
                    z + h / 2,
                    z + h / 2,
                    z + h / 2,
                ],
            ]
        )
        return box8.T