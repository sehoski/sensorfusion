import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class SensorFusionVisualizationNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_visualization_node')
        
        self.bridge = CvBridge()
        
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
        
        # 발행자 설정
        self.fusion_pub = self.create_publisher(Image, '/fusion/image', 10)
        
        self.latest_image = None
        self.latest_radar_points = None
        
        # 캘리브레이션 파라미터 (예시 값, 실제 값으로 대체 필요)
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.dist_coeffs = np.zeros((5,1))
        self.rvec = np.zeros((3,1))
        self.tvec = np.array([0, 0, 0.07]).reshape((3,1))

    def camera_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_and_publish()

    def radar_callback(self, msg):
        self.latest_radar_points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        self.process_and_publish()

    def process_and_publish(self):
        if self.latest_image is None or self.latest_radar_points is None:
            return

        # 반사체 검출
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(self.latest_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 반사체 중심점 계산
            reflector_center = np.array([x + w/2, y + h/2])

            # 레이더 포인트 투영
            radar_points_3d = np.array([[p[0], p[1], p[2]] for p in self.latest_radar_points])
            radar_points_2d, _ = cv2.projectPoints(radar_points_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)

            # RMSE 계산을 위한 오차 누적
            squared_errors = []

            # 바운딩 박스 내의 포인트만 그리기 및 오차 계산
            for point in radar_points_2d.reshape(-1, 2):
                px, py = point
                if x < px < x+w and y < py < y+h:
                    cv2.circle(self.latest_image, (int(px), int(py)), 3, (0, 0, 255), -1)
                    error = np.linalg.norm(point - reflector_center)
                    squared_errors.append(error**2)

            # RMSE 계산
            if squared_errors:
                rmse = np.sqrt(np.mean(squared_errors))
                self.get_logger().info(f'RMSE: {rmse:.4f} pixels')
            else:
                self.get_logger().info('No radar points detected within the bounding box')

        # 결과 이미지 발행
        fusion_msg = self.bridge.cv2_to_imgmsg(self.latest_image, "bgr8")
        self.fusion_pub.publish(fusion_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionVisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
