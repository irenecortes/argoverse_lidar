import sys
ros_path = '/opt/ros/lunar/lib/python2.7/dist-packages'

if ros_path in sys.path:
    print(sys.path)
    sys.path.remove(ros_path)

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import argoverse.visualization.visualization_utils as viz_util
import cv2

sys.path.append('/opt/ros/lunar/lib/python2.7/dist-packages')
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
import numpy as np
import ros_numpy

vlp32_planes = {31: -25.,
                30: -15.639,
                29: -11.31,
                28: -8.843,
                27: -7.254,
                26: -6.148,
                25: -5.333,
                24: -4.667,
                23: -4.,
                22: -3.667,
                21: -3.333,
                20: -3.,
                19: -2.667,
                18: -2.333,
                17: -2.,
                16: -1.667,
                15: -1.333,
                14: -1.,
                13: -0.667,
                12: -0.333,
                11: 0.,
                10: 0.333,
                9:  0.667,
                8:  1.,
                7:  1.333,
                6:  1.667,
                5:  2.333,
                4:  3.333,
                3:  4.667,
                2:  7.,
                1:  10.333,
                0:  15.}

from scipy.spatial.transform import Rotation

def separate_pc(pc, tf_up, tf_down):

    pc_points = np.ones((len(pc), 4))
    pc_points[:,0:3] = pc[:,0:3]

    pc_up_tf = np.dot(np.linalg.inv(tf_up),  pc_points.transpose())
    pc_down_tf = np.dot(np.linalg.inv(tf_down), pc_points.transpose())

    pc_up_dis = np.sqrt(pc_up_tf[0,:]**2 + pc_up_tf[1,:]**2 + pc_up_tf[2,:]**2)
    pc_up_omega = np.arcsin(pc_up_tf[2,:]/pc_up_dis) * 180 / np.pi

    pc_down_dis = np.sqrt(pc_down_tf[0,:]**2 + pc_down_tf[1,:]**2 + pc_down_tf[2,:]**2)
    pc_down_omega = np.arcsin(pc_down_tf[2,:]/pc_down_dis) * 180 / np.pi

    pc_angles = np.array([vlp32_planes[pc[i,4]] for i in range(0, len(pc))])

    pc_up_xyz = pc_up_tf[:, (np.fabs(pc_up_omega - pc_angles) < 2.0) * (np.fabs(pc_down_omega - pc_angles) > 2.0)]
    pc_down_xyz = pc_down_tf[:, (np.fabs(pc_up_omega - pc_angles) > 2.0) * (np.fabs(pc_down_omega -  pc_angles) < 2.0)]

    pc_up = np.zeros((pc_up_xyz.shape[1], 5))
    pc_down = np.zeros((pc_down_xyz.shape[1], 5))

    pc_up[:,0:3] = pc_up_xyz.transpose()[:,0:3]
    pc_down[:,0:3] = pc_down_xyz.transpose()[:,0:3]
    pc_up[:,3:5] = pc[(np.fabs(pc_up_omega - pc_angles) < 2.0) * (np.fabs(pc_down_omega - pc_angles) > 2.0), 3:5]
    pc_down[:,3:5] = pc[(np.fabs(pc_up_omega - pc_angles) > 2.0) * (np.fabs(pc_down_omega -  pc_angles) < 2.0), 3:5]


    return pc_up, pc_down

def main():

    rospy.init_node('publish_argoverse_velodyne', anonymous=True)
    velo_pub_up = rospy.Publisher("velodyne_points_up", PointCloud2, queue_size=1)
    velo_pub_down = rospy.Publisher("velodyne_points_down", PointCloud2, queue_size=1)

    root_dir =  'ARGOVERSE_ROOT/argoverse-api/argoverse-tracking/sample/'
    argoverse_loader = ArgoverseTrackingLoader(root_dir)

    log_id = 'c6911883-1843-3727-8eaa-41dc8cda8993'
    r = rospy.Rate(10)
    
    tf_down_lidar_rot = Rotation.from_quat([-0.9940207559208627, -0.10919018413803058, -0.00041138986312043766, -0.00026691721622102603])
    tf_down_lidar_tr = [1.3533224859271054, -0.0009818949950377448, 1.4830535977952262]

    tf_down_lidar = np.eye(4)
    tf_down_lidar[0:3,0:3] = tf_down_lidar_rot.as_dcm()
    tf_down_lidar[0:3,3] = tf_down_lidar_tr

    tf_up_lidar_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    tf_up_lidar_tr = [1.35018, 0.0, 1.59042]
    tf_up_lidar = np.eye(4)
    tf_up_lidar[0:3,0:3] = tf_up_lidar_rot.as_dcm()
    tf_up_lidar[0:3,3] = tf_up_lidar_tr

    for idx in range(0, argoverse_loader.lidar_count):
        while not rospy.is_shutdown():

            argoverse_data = argoverse_loader.get(log_id)
            pc = argoverse_data.get_lidar_ring(idx)

            # pc_ring_0 = pc[pc[:,4]==15]

            pc_up, pc_down = separate_pc(pc, tf_up_lidar, tf_down_lidar)


            data = np.zeros(pc_up.shape[0], dtype=[
              ('x', np.float32),
              ('y', np.float32),
              ('z', np.float32),
              ('intensity', np.uint8),
              ('ring', np.uint8),
            ])
            data['x'] = pc_up[:,0]
            data['y'] = pc_up[:,1]
            data['z'] = pc_up[:,2]
            data['intensity'] = pc_up[:,3]
            data['ring'] = pc_up[:,4]

            msg = ros_numpy.msgify(PointCloud2, data)
            msg.header.frame_id = 'velo_up'
            velo_pub_up.publish(msg)


            data = np.zeros(pc_down.shape[0], dtype=[
              ('x', np.float32),
              ('y', np.float32),
              ('z', np.float32),
              ('intensity', np.uint8),
              ('ring', np.uint8),
            ])

            data['x'] = pc_down[:,0]
            data['y'] = pc_down[:,1]
            data['z'] = pc_down[:,2]
            data['intensity'] = pc_down[:,3]
            data['ring'] = pc_down[:,4]

            msg = ros_numpy.msgify(PointCloud2, data)
            msg.header.frame_id = 'velo_down'
            velo_pub_down.publish(msg)
            # print(pc.shape)
            r.sleep()

if __name__ == '__main__':
    main()
