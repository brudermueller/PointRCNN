import numpy as np 


class CustomObject3d(object):
    def __init__(self, bbox_array): 
        #bbox: (array of length 7) [x, y, z, h, w, l, rz]
        self.pos = bbox[0:3]
        self.h, self.w, self.l = float(bbox[3]), float(bbox[4]), float(bbox[5])
        self.rz =  bbox[6]  # rotation angle around z-axis (instead of y as in camera coord.)
        self.dis_to_sensor = np.linalg.norm(self.pos)
        # According to KITTI definition
        self.cls_type = 'Pedestrian'
        self.cls_id = 2

    def generate_corners3d(self):
        """
        Generate corners3d representation for this object
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1
        :return corners_3d: (8, 3) corners of oriented box3d in Velodyne coord.
        """
        l, h, w = self.l, self.h, self.w
        # careful: width, length and height have been differently defined than in KITTI
        x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]        
        y_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
        z_corners = [0, 0, 0, h / 2, h / 2, h / 2]

        # rotation now defined in Velodyne coords. -> around z-axis => yaw rot. 
        R = np.array([[np.cos(self.rz), -np,sin(self.rz), 0],
                      [np.sin(self.rz), np.cos(self.rz), 0],
                      [0, 0, 1]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        # transpose and rotate around orientation angle 
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def generate_unoriented_bbox(self): 
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