import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ExternalCalibrationNode(Node):
    def __init__(self):
        super().__init__('external_calibration_node')
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(Image, 'camera/image_raw', self.camera_callback, 10)
        self.radar_sub = self.create_subscription(PointCloud2, 'radar/points', self.radar_callback, 10)
        self.camera_data = []
        self.radar_data = []
        self.matching_points = []

    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        reflector_position = self.detect_reflector(cv_image)
        if reflector_position is not None:
            self.camera_data.append(reflector_position)
            self.get_logger().info(f'Reflector detected in camera at {reflector_position}')

    def radar_callback(self, msg):
        radar_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"))))
        valid_points = radar_points[radar_points[:, 3] > 15]  # RCS 필터링
        if valid_points.shape[0] > 0:
            max_reflection_point = valid_points[np.argmax(valid_points[:, 3])]
            self.radar_data.append(max_reflection_point[:3])
            self.get_logger().info(f'Reflector detected in radar at {max_reflection_point[:3]}')

    def detect_reflector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour = max(contours, key=cv2.contourArea, default=None)
        
        if max_contour is not None:
            M = cv2.moments(max_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return np.array([cx, cy, 0.0])
        return None

    def match_points(self):
        min_len = min(len(self.camera_data), len(self.radar_data))
        if min_len >= 3:
            self.matching_points = list(zip(self.camera_data[:min_len], self.radar_data[:min_len]))
        else:
            self.get_logger().warn('Not enough matching points for calibration.')

    def calibrate(self):
        self.match_points()
        if len(self.matching_points) < 3:
            self.get_logger().warn('Not enough matching points for calibration. Gathering more data...')
            return

        def objective_function(params):
            R_matrix = R.from_rotvec(params[:3]).as_matrix()
            t_vector = params[3:]
            errors = []
            for camera_point, radar_point in self.matching_points:
                transformed_point = R_matrix @ np.array(radar_point) + t_vector
                error = np.linalg.norm(transformed_point - camera_point)
                errors.append(error)
            return errors

        initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.07])

        result = least_squares(objective_function, initial_guess, method='lm', ftol=1e-8, xtol=1e-8, max_nfev=1000)

        R_calibrated = R.from_rotvec(result.x[:3]).as_matrix()
        t_calibrated = result.x[3:]

        self.get_logger().info(f'Calibration complete. Rotation Matrix:\n{R_calibrated}')
        self.get_logger().info(f'Translation Vector: {t_calibrated}')

        self.visualize_calibration(R_calibrated, t_calibrated)

        self.camera_data.clear()
        self.radar_data.clear()

        return R_calibrated, t_calibrated

    def visualize_calibration(self, R_calibrated, t_calibrated):
        fig = plt.figure(figsize=(15, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        camera_points = np.array(self.camera_data)
        radar_points = np.array(self.radar_data)
        transformed_radar_points = np.dot(R_calibrated, radar_points.T).T + t_calibrated

        ax1.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], c='b', marker='o', label='Camera Points')
        ax1.scatter(radar_points[:, 0], radar_points[:, 1], radar_points[:, 2], c='r', marker='^', label='Radar Points')
        ax1.scatter(transformed_radar_points[:, 0], transformed_radar_points[:, 1], transformed_radar_points[:, 2], c='g', marker='x', label='Transformed Radar Points')

        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')
        ax1.legend()

        # 2D plot (top-down view)
        ax2 = fig.add_subplot(122)
        ax2.scatter(camera_points[:, 0], camera_points[:, 1], c='b', marker='o', label='Camera Points')
        ax2.scatter(radar_points[:, 0], radar_points[:, 1], c='r', marker='^', label='Radar Points')
        ax2.scatter(transformed_radar_points[:, 0], transformed_radar_points[:, 1], c='g', marker='x', label='Transformed Radar Points')
        
        for i in range(len(radar_points)):
            ax2.arrow(0, 0, radar_points[i, 0], radar_points[i, 1], color='r', alpha=0.5, width=0.001)
            ax2.annotate(f'{np.linalg.norm(radar_points[i]):.2f}m', (radar_points[i, 0], radar_points[i, 1]))

        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.legend()
        ax2.axis('equal')

        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            if len(node.camera_data) >= 20 and len(node.radar_data) >= 20:
                node.calibrate()
                break
            node.get_logger().info('Waiting for enough data to perform calibration...')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
