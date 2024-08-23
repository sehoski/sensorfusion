import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ExternalCalibrationNode(Node):
    def __init__(self):
        super().__init__('external_calibration_node')
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(Image, 'camera/image_raw', self.camera_callback, 10)
        self.radar_sub = self.create_subscription(Point, 'radar/transformed_point', self.radar_callback, 10)
        self.camera_data = []
        self.radar_data = []
        self.matching_points = []
        self.calibration_done = False
        self.create_timer(1.0, self.check_calibration)  # 1초마다 캘리브레이션 상태 확인

    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        reflector_position = self.detect_reflector(cv_image)
        if reflector_position is not None:
            self.camera_data.append(reflector_position)
            self.get_logger().info(f'Reflector detected in camera at {reflector_position}')

    def radar_callback(self, msg):
        radar_point = np.array([msg.x, msg.y, msg.z])
        self.radar_data.append(radar_point)
        self.get_logger().info(f'Radar point received: {radar_point}')

    def check_calibration(self):
        self.get_logger().info(f"Current data points: Camera {len(self.camera_data)}, Radar {len(self.radar_data)}")
        if not self.calibration_done and len(self.radar_data) >= 20 and len(self.camera_data) >= 20:
            self.get_logger().info("Starting calibration...")
            R_calibrated, t_calibrated = self.calibrate()
            if R_calibrated is not None and t_calibrated is not None:
                self.calibration_done = True
                self.get_logger().info("Calibration completed. Displaying graph...")
                self.visualize_calibration(R_calibrated, t_calibrated)
                plt.show(block=False)
                plt.pause(0.001)  # 그래프 업데이트를 위한 잠시 멈춤
                self.get_logger().info("Graph displayed. Press Ctrl+C to exit.")

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

    def rotation_matrix(self, angles):
        x, y, z = angles
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def objective_function(self, params):
        angles, t_vector = params[:3], params[3:]
        R_matrix = self.rotation_matrix(angles)
        errors = []
        for camera_point, radar_point in self.matching_points:
            transformed_point = R_matrix @ np.array(radar_point) + t_vector
            error = np.linalg.norm(transformed_point - camera_point)
            errors.append(error)
        return np.mean(errors)

    def gradient_descent(self, initial_guess, learning_rate=0.01, num_iterations=1000):
        params = initial_guess
        for _ in range(num_iterations):
            grad = np.zeros_like(params)
            for i in range(len(params)):
                h = np.zeros_like(params)
                h[i] = 1e-4  # small step
                grad[i] = (self.objective_function(params + h) - self.objective_function(params - h)) / (2 * h[i])
            params -= learning_rate * grad
        return params

    def calibrate(self):
        self.match_points()
        if len(self.matching_points) < 3:
            self.get_logger().warn('Not enough matching points for calibration. Gathering more data...')
            return None, None

        initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.07])
        result = self.gradient_descent(initial_guess)

        R_calibrated = self.rotation_matrix(result[:3])
        t_calibrated = result[3:]

        self.get_logger().info(f'Calibration complete. Rotation Matrix:\n{R_calibrated}')
        self.get_logger().info(f'Translation Vector: {t_calibrated}')

        return R_calibrated, t_calibrated

    def visualize_calibration(self, R_calibrated, t_calibrated):
        plt.figure(figsize=(15, 5))
        
        # 3D plot
        ax1 = plt.subplot(121, projection='3d')
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
        ax2 = plt.subplot(122)
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

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        try:
            rclpy.try_shutdown()
        except:
            print("ROS 2 context already shut down.")

if __name__ == '__main__':
    main()
