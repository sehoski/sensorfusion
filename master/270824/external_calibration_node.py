import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.optimize import least_squares
import tf2_ros
import tf2_geometry_msgs
import sensor_msgs_py.point_cloud2 as pc2

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
        
        # 타이머 설정 (10초마다 캘리브레이션 수행)
        self.timer = self.create_timer(10.0, self.calibration_timer_callback)

    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        reflector_center = self.detect_reflector(cv_image)
        if reflector_center is not None:
            self.camera_data.append(reflector_center)
            self.get_logger().info(f'Detected reflector in camera image at {reflector_center}')

    def radar_callback(self, msg):
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        reflector_center = self.extract_reflector_center(points)
        if reflector_center is not None:
            self.radar_data.append(reflector_center)
            self.get_logger().info(f'Detected reflector in radar data at {reflector_center}')

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
        # 실제 구현에서는 레이더 포인트 클라우드에서 반사체 중심을 찾아야 합니다.
        # 여기서는 간단히 모든 포인트의 평균을 계산합니다.
        points_array = np.array(list(points))
        if len(points_array) > 0:
            return np.mean(points_array, axis=0)
        return None

    def calibration_timer_callback(self):
        if len(self.camera_data) > 10 and len(self.radar_data) > 10:
            self.perform_calibration()
            self.publish_calibration_result()
            # 캘리브레이션 후 데이터 초기화
            self.camera_data = []
            self.radar_data = []

    def perform_calibration(self):
        camera_points = np.array(self.camera_data)
        radar_points = np.array(self.radar_data)
        
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
        # 이전 코드와 동일

    def project_points(self, points_3d, R, t):
        # 이전 코드와 동일, 단 실제 카메라 파라미터 사용 필요

    def publish_calibration_result(self):
        if self.calib_params is not None:
            transform_msg = Transform()
            transform_msg.translation.x = self.calib_params[0]
            transform_msg.translation.y = self.calib_params[1]
            transform_msg.translation.z = self.calib_params[2]
            
            # 오일러 각을 쿼터니언으로 변환
            quat = tf2_geometry_msgs.transformations.quaternion_from_euler(
                self.calib_params[3], self.calib_params[4], self.calib_params[5])
            transform_msg.rotation.x = quat[0]
            transform_msg.rotation.y = quat[1]
            transform_msg.rotation.z = quat[2]
            transform_msg.rotation.w = quat[3]
            
            self.calib_pub.publish(transform_msg)
            self.get_logger().info('Published calibration result')

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
