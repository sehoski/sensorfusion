import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.optimize import least_squares
import tf2_ros
import transforms3d  # tf-transformations 대신 사용
import sensor_msgs_py.point_cloud2 as pc2
import json

class ExternalCalibrationNode(Node):
    def __init__(self):
        super().__init__('external_calibration_node')
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 구독자 설정
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10)
        self.radar_sub = self.create_subscription(
            PointCloud2,
            '/radar/points',
            self.radar_callback,
            10)
        
        # 퍼블리셔 설정
        self.calib_pub = self.create_publisher(Transform, '/camera_radar_transform', 10)
        
        # 데이터 저장
        self.camera_data = []
        self.radar_data = []
        
        # 캘리브레이션 파라미터
        self.calib_params = None
        
        # 카메라 내부 파라미터 (초기화)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 타이머 설정 (10초마다 캘리브레이션 수행)
        self.timer = self.create_timer(10.0, self.calibration_timer_callback)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Received initial camera info')

    def camera_callback(self, msg):
        if self.camera_matrix is None:
            return
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        reflector_center = self.detect_reflector(cv_image)
        if reflector_center is not None:
            self.camera_data.append(reflector_center)
            self.get_logger().debug(f'Detected reflector in camera image at {reflector_center}')

    def radar_callback(self, msg):
        points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        reflector_center = self.extract_reflector_center(points)
        if reflector_center is not None:
            self.radar_data.append(reflector_center[:3])  # x, y, z만 저장
            self.get_logger().info(f'Detected reflector in radar data at {reflector_center[:3]}')

    def detect_reflector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return np.array([cx, cy])
        return None

    def extract_reflector_center(self, points):
        points_list = list(points)
        self.get_logger().debug(f"Received {len(points_list)} points")
        if len(points_list) > 0:
            points_array = np.array(points_list)
            self.get_logger().debug(f"Points array shape: {points_array.shape}")
            
            if len(points_array) > 0:
                # 각 필드별로 평균을 계산
                mean_x = np.mean(points_array['x'])
                mean_y = np.mean(points_array['y'])
                mean_z = np.mean(points_array['z'])
                mean_intensity = np.mean(points_array['intensity'])
                
                reflector_center = np.array([mean_x, mean_y, mean_z, mean_intensity])
                return reflector_center
        else:
            self.get_logger().warn('No radar points received')
        return None

    def calibration_timer_callback(self):
        if self.camera_matrix is None:
            self.get_logger().warn('Camera info not received yet. Skipping calibration.')
            return
        
        if len(self.camera_data) > 10 and len(self.radar_data) > 10:
            self.get_logger().info(f'Performing calibration with {len(self.camera_data)} camera points and {len(self.radar_data)} radar points')
            self.perform_calibration()
            self.publish_calibration_result()
            self.save_calibration_result()
            # 캘리브레이션 후 데이터 초기화
            self.camera_data = []
            self.radar_data = []
        else:
            self.get_logger().info(f'Not enough data for calibration. Camera points: {len(self.camera_data)}, Radar points: {len(self.radar_data)}')

    def perform_calibration(self):
        camera_points = np.array(self.camera_data)
        radar_points = np.array(self.radar_data)
        
        # 두 포인트 집합의 최소 개수에 맞추기
        min_points = min(len(camera_points), len(radar_points))
        if len(camera_points) != len(radar_points):
            self.get_logger().warn(f"Different number of points: Camera {len(camera_points)}, Radar {len(radar_points)}. Using {min_points} points for calibration.")
        
        camera_points = camera_points[:min_points]
        radar_points = radar_points[:min_points]
        
        # 초기 추정값 (카메라가 레이더보다 7cm 위에 있다고 가정)
        initial_guess = np.array([0, 0, 0.07, 0, 0, 0])  # [x, y, z, roll, pitch, yaw]
        
        result = least_squares(self.projection_error, initial_guess, 
                               args=(camera_points, radar_points))
        
        self.calib_params = result.x
        rmse = np.sqrt(np.mean(result.fun**2))
        self.get_logger().info(f'Calibration completed. RMSE: {rmse:.4f}')

    def projection_error(self, params, camera_points, radar_points):
        R = self.euler_to_rotmat(params[3:])
        t = params[:3]
        
        projected_points = self.project_points(radar_points, R, t)
        errors = camera_points - projected_points
        return errors.ravel()

    def euler_to_rotmat(self, euler_angles):
        Rx = transforms3d.euler.euler2mat(euler_angles[0], 0, 0)
        Ry = transforms3d.euler.euler2mat(0, euler_angles[1], 0)
        Rz = transforms3d.euler.euler2mat(0, 0, euler_angles[2])
        return np.dot(Rz, np.dot(Ry, Rx))

    def project_points(self, points_3d, R, t):
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        points_cam = np.dot(R, points_3d.T).T + t
        
        points_2d = np.zeros((points_cam.shape[0], 2))
        points_2d[:, 0] = fx * points_cam[:, 0] / points_cam[:, 2] + cx
        points_2d[:, 1] = fy * points_cam[:, 1] / points_cam[:, 2] + cy
        
        return points_2d

    def publish_calibration_result(self):
        if self.calib_params is not None:
            transform_msg = Transform()
            transform_msg.translation.x = self.calib_params[0]
            transform_msg.translation.y = self.calib_params[1]
            transform_msg.translation.z = self.calib_params[2]
            
            quat = transforms3d.euler.euler2quat(
                self.calib_params[3], self.calib_params[4], self.calib_params[5])
            transform_msg.rotation.x = quat[1]
            transform_msg.rotation.y = quat[2]
            transform_msg.rotation.z = quat[3]
            transform_msg.rotation.w = quat[0]
            
            self.calib_pub.publish(transform_msg)
            self.get_logger().info('Published calibration result')

    def save_calibration_result(self):
        if self.calib_params is not None:
            calib_data = {
                'translation': self.calib_params[:3].tolist(),
                'rotation': self.calib_params[3:].tolist()
            }
            with open('calibration_result.json', 'w') as f:
                json.dump(calib_data, f)
            self.get_logger().info('Saved calibration result to file')

    def load_calibration_result(self):
        try:
            with open('calibration_result.json', 'r') as f:
                calib_data = json.load(f)
            self.calib_params = np.array(calib_data['translation'] + calib_data['rotation'])
            self.get_logger().info('Loaded calibration result from file')
        except FileNotFoundError:
            self.get_logger().warn('No calibration file found. Starting with default parameters.')

    def destroy_node(self):
        self.save_calibration_result()  # 노드 종료 시 결과 저장
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()
    node.load_calibration_result()  # 노드 시작 시 이전 결과 로드
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

