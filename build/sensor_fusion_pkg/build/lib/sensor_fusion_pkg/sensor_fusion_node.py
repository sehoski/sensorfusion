import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
<<<<<<< HEAD
=======
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
>>>>>>> 412fa61dba708d13d55ef78287d51cd050c5ef63

class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('time_sync_node')

        self.get_logger().info('Initializing TimeSyncNode')

        self.camera_sub = Subscriber(self, Image, 'camera/image_raw')
        self.radar_sub = Subscriber(self, PointCloud2, 'radar/points')

<<<<<<< HEAD
        # 슬로프 값을 0.1로 조정하여 더 엄격한 동기화 시도
=======
>>>>>>> 412fa61dba708d13d55ef78287d51cd050c5ef63
        self.ts = ApproximateTimeSynchronizer(
            [self.camera_sub, self.radar_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        
<<<<<<< HEAD
        self.get_logger().info('Subscribers initialized and ApproximateTimeSynchronizer set')

    def callback(self, camera_data, radar_data):
        self.get_logger().info('Callback function called')  # 콜백 호출 확인

        camera_time = camera_data.header.stamp
        radar_time = radar_data.header.stamp

        # Log the timestamps
        self.get_logger().info(f'Camera time: {camera_time.sec}.{camera_time.nanosec}')
        self.get_logger().info(f'Radar time: {radar_time.sec}.{radar_time.nanosec}')
        
        # Calculate time difference
        time_diff_sec = abs(camera_time.sec - radar_time.sec)
        time_diff_nsec = abs(camera_time.nanosec - radar_time.nanosec) / 1e9
        total_time_diff = time_diff_sec + time_diff_nsec
        
        # Log the time difference
        self.get_logger().info(f'Time difference: {total_time_diff} seconds')

        # Process synchronized messages
        self.get_logger().info('Synchronized messages received')
        cv_image = self.bridge.imgmsg_to_cv2(camera_data, desired_encoding='bgr8')
        self.get_logger().info(f'Received radar data with {len(radar_data.data)} points')
        cv2.imshow('Camera', cv_image)
        cv2.waitKey(1)

=======
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

>>>>>>> 412fa61dba708d13d55ef78287d51cd050c5ef63
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

