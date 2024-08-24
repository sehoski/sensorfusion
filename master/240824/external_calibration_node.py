import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import message_filters

class ExternalCalibrationNode(Node):
    def __init__(self):
        super().__init__('external_calibration_node')
        self.bridge = CvBridge()
        
        # 시간 동기화를 위한 메시지 필터 사용
        self.camera_sub = message_filters.Subscriber(self, Image, 'camera/image_raw')
        self.radar_sub = message_filters.Subscriber(self, PointStamped, 'radar/transformed_point')
        
        # 0.1초의 시간 허용 오차로 메시지 동기화
        ts = message_filters.ApproximateTimeSynchronizer([self.camera_sub, self.radar_sub], 10, 0.1)
        ts.registerCallback(self.synchronized_callback)
        
        # 메모리 관리를 위해 최대 길이가 제한된 deque 사용
        self.camera_data = deque(maxlen=100)
        self.radar_data = deque(maxlen=100)
        self.matching_points = deque(maxlen=100)
        
        # 반사판 감지 임계값을 ROS 파라미터로 설정
        self.declare_parameter('reflector_threshold', 240)
        self.reflector_threshold = self.get_parameter('reflector_threshold').value

    def synchronized_callback(self, camera_msg, radar_msg):
        # 카메라 이미지를 OpenCV 형식으로 변환
        cv_image = self.bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        reflector_position = self.detect_reflector(cv_image)
        
        if reflector_position is not None:
            # 레이더 포인트를 numpy 배열로 변환
            radar_point = np.array([radar_msg.point.x, radar_msg.point.y, radar_msg.point.z])
            self.camera_data.append(reflector_position)
            self.radar_data.append(radar_point)
            self.matching_points.append((reflector_position, radar_point))
            
            self.get_logger().info(f'매칭된 포인트: 카메라 {reflector_position}, 레이더 {radar_point}')
            
            # 충분한 데이터가 모이면 캘리브레이션 수행
            if len(self.matching_points) >= 20:
                self.calibrate()

    def detect_reflector(self, image):
        # 이미지를 그레이스케일로 변환하고 이진화
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.reflector_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.get_logger().warn('이미지에서 윤곽선을 찾을 수 없습니다')
            return None
        
        # 가장 큰 윤곽선을 반사판으로 간주
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        
        if M['m00'] == 0:
            self.get_logger().warn('유효하지 않은 윤곽선 모멘트')
            return None
        
        # 반사판의 중심 좌표 계산
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy, 0.0])

    def calibrate(self):
        if len(self.matching_points) < 3:
            self.get_logger().warn('캘리브레이션을 위한 충분한 매칭 포인트가 없습니다. 더 많은 데이터를 수집 중...')
            return

        def objective_function(params):
            # 회전 행렬과 변환 벡터 계산
            R_matrix = R.from_rotvec(params[:3]).as_matrix()
            t_vector = params[3:]
            errors = []
            for camera_point, radar_point in self.matching_points:
                # 레이더 포인트를 카메라 좌표계로 변환
                transformed_point = R_matrix @ np.array(radar_point) + t_vector
                error = np.linalg.norm(transformed_point - camera_point)
                errors.append(error)
            return errors

        # 최적화 초기 추정치
        initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.07])
        # Levenberg-Marquardt 알고리즘을 사용한 최적화
        result = least_squares(objective_function, initial_guess, method='lm', ftol=1e-8, xtol=1e-8, max_nfev=1000)

        # 최적화 결과로부터 회전 행렬과 변환 벡터 추출
        R_calibrated = R.from_rotvec(result.x[:3]).as_matrix()
        t_calibrated = result.x[3:]

        # RMSE(Root Mean Square Error) 계산
        rmse = np.sqrt(np.mean(np.square(result.fun)))
        self.get_logger().info(f'캘리브레이션 완료. RMSE: {rmse}')
        self.get_logger().info(f'회전 행렬:\n{R_calibrated}')
        self.get_logger().info(f'변환 벡터: {t_calibrated}')

        # 결과 시각화
        self.visualize_calibration(R_calibrated, t_calibrated)

    def visualize_calibration(self, R_calibrated, t_calibrated):
        fig = plt.figure(figsize=(15, 5))
        
        # 3D 플롯
        ax1 = fig.add_subplot(121, projection='3d')
        camera_points = np.array([point for point, _ in self.matching_points])
        radar_points = np.array([point for _, point in self.matching_points])
        transformed_radar_points = np.dot(R_calibrated, radar_points.T).T + t_calibrated

        ax1.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], c='b', marker='o', label='카메라 포인트')
        ax1.scatter(radar_points[:, 0], radar_points[:, 1], radar_points[:, 2], c='r', marker='^', label='레이더 포인트')
        ax1.scatter(transformed_radar_points[:, 0], transformed_radar_points[:, 1], transformed_radar_points[:, 2], c='g', marker='x', label='변환된 레이더 포인트')

        ax1.set_xlabel('X 축')
        ax1.set_ylabel('Y 축')
        ax1.set_zlabel('Z 축')
        ax1.legend()

        # 2D 플롯 (탑뷰)
        ax2 = fig.add_subplot(122)
        ax2.scatter(camera_points[:, 0], camera_points[:, 1], c='b', marker='o', label='카메라 포인트')
        ax2.scatter(radar_points[:, 0], radar_points[:, 1], c='r', marker='^', label='레이더 포인트')
        ax2.scatter(transformed_radar_points[:, 0], transformed_radar_points[:, 1], c='g', marker='x', label='변환된 레이더 포인트')
        
        for i in range(len(radar_points)):
            ax2.arrow(0, 0, radar_points[i, 0], radar_points[i, 1], color='r', alpha=0.5, width=0.001)
            ax2.annotate(f'{np.linalg.norm(radar_points[i]):.2f}m', (radar_points[i, 0], radar_points[i, 1]))

        ax2.set_xlabel('X 축')
        ax2.set_ylabel('Y 축')
        ax2.legend()
        ax2.axis('equal')

        plt.tight_layout()
        plt.savefig('calibration_result.png')
        self.get_logger().info('캘리브레이션 시각화 결과가 calibration_result.png로 저장되었습니다')

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
