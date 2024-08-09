import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('time_sync_node')

        self.get_logger().info('Initializing TimeSyncNode')

        self.camera_sub = Subscriber(self, Image, 'camera/image_raw')
        self.radar_sub = Subscriber(self, PointCloud2, 'radar/points')

        self.ts = ApproximateTimeSynchronizer(
            [self.camera_sub, self.radar_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        
        # 캘리브레이션 행렬 로드
        calibration_data = np.load('/home/seho/ros2_ws/src/my_camera_pkg/my_camera_pkg/calibration_result.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.rotation_matrix = calibration_data['rotation_matrix']
        self.translation_vector = calibration_data['translation_vector']

        self.get_logger().info('Subscribers initialized and ApproximateTimeSynchronizer set')

    def callback(self, camera_data, radar_data):
        self.get_logger().info('Callback function called')

        cv_image = self.bridge.imgmsg_to_cv2(camera_data, desired_encoding='bgr8')
        undistorted_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)

        radar_points = list(pc2.read_points(radar_data, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        radar_points = np.array([(x, y, z) for x, y, z, rgb in radar_points])

        if radar_points.size == 0:
            return

        radar_points_homogeneous = np.hstack((radar_points, np.ones((radar_points.shape[0], 1))))
        transformed_points = radar_points_homogeneous.dot(self.rotation_matrix.T) + self.translation_vector.T

        for point in transformed_points:
            x, y, z = point[:3]
            pixel_coords = self.project_to_image_plane(x, y, z)
            if 0 <= pixel_coords[0] < undistorted_image.shape[1] and 0 <= pixel_coords[1] < undistorted_image.shape[0]:
                cv2.circle(undistorted_image, tuple(pixel_coords.astype(int)), 3, (0, 0, 255), -1)

        cv2.imshow('Fused Image', undistorted_image)
        cv2.waitKey(1)

    def project_to_image_plane(self, x, y, z):
        point_3d = np.array([x, y, z]).reshape((3, 1))
        projected_point = self.camera_matrix.dot(point_3d)
        projected_point /= projected_point[2]
        return projected_point[:2].flatten()

def main(args=None):
    rclpy.init(args=args)
    time_sync_node = TimeSyncNode()
    try:
        rclpy.spin(time_sync_node)
    except KeyboardInterrupt:
        pass
    finally:
        time_sync_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

