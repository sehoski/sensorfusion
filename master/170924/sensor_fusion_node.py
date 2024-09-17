import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2
import json
import os

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # 카메라 토픽 구독 (객체 탐지 결과를 포함)
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        # 레이더 토픽 구독 (PointCloud2 메시지 형태로 레이더 데이터 수신)
        self.radar_sub = self.create_subscription(
            PointCloud2,
            '/radar/points',
            self.radar_callback,
            10
        )

        # OpenCV와 ROS 이미지 변환용 CvBridge
        self.bridge = CvBridge()

        # 카메라에서 탐지한 객체 저장용
        self.detected_objects = []

        # 레이더 데이터 저장용
        self.radar_points = []

        # 외부 캘리브레이션 결과 불러오기
        self.load_calibration_data()

    def load_calibration_data(self):
        # 캘리브레이션 파일 경로
        calibration_file_path = '/home/ros2_ws/src/sensor_fusion_pkg/calibration_result_20240827-133738.json'
        
        # 외부 캘리브레이션 결과값 불러오기
        try:
            with open(calibration_file_path, 'r') as f:
                calibration_data = json.load(f)
                self.rotation_matrix = np.array(calibration_data['rotation_matrix'])
                self.translation_vector = np.array(calibration_data['translation_vector'])
                self.camera_matrix = np.array(calibration_data['camera_matrix'])
                self.get_logger().info('Calibration data loaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration data: {e}')

    def transform_radar_point(self, radar_point):
        # 레이더 포인트를 카메라 좌표계로 변환
        radar_coords = np.array([[radar_point[0]], [radar_point[1]], [radar_point[2]]])  # 레이더 포인트 (x, y, z)
        transformed_coords = np.dot(self.rotation_matrix, radar_coords) + self.translation_vector  # 회전 및 변환 적용
        return transformed_coords.flatten()  # 변환된 (x, y, z) 좌표 반환

    def project_to_2d(self, radar_point_3d):
        # 3D 좌표를 카메라 이미지 평면으로 투영
        radar_coords_3d = np.array([[radar_point_3d[0]], [radar_point_3d[1]], [radar_point_3d[2]], [1]])
        image_coords = np.dot(self.camera_matrix, radar_coords_3d)
        u = image_coords[0] / image_coords[2]  # x 좌표 (화면 상)
        v = image_coords[1] / image_coords[2]  # y 좌표 (화면 상)
        return int(u), int(v)  # 2D 좌표 반환

    def camera_callback(self, msg):
        try:
            # CvBridge를 사용하여 ROS Image 메시지를 OpenCV 이미지로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 객체 정보는 self.detected_objects에 저장된다고 가정
            self.detected_objects = [
                {'class': 'car', 'bbox': [50, 50, 100, 100]},  # 예시 객체
            ]

            # 융합 데이터를 오버레이하여 시각적으로 표시
            self.display_fusion_data(frame)

        except Exception as e:
            self.get_logger().error(f"Error in camera_callback: {e}")

    def radar_callback(self, msg):
        try:
            # PointCloud2 메시지에서 데이터를 추출
            radar_points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "velocity", "intensity"), skip_nans=True))
            
            # 레이더 포인트를 카메라 좌표계로 변환
            self.radar_points = [self.transform_radar_point(point) for point in radar_points]
        except Exception as e:
            self.get_logger().error(f"Error in radar_callback: {e}")

    def match_radar_to_object(self, obj):
        # 바운딩 박스의 중심점 계산
        x, y, w, h = obj['bbox']
        obj_center_x = x + w / 2
        obj_center_y = y + h / 2

        # 레이더 포인트와 객체 중심의 2D 거리 계산 후, 가장 가까운 레이더 포인트를 매칭
        closest_point = None
        min_distance = float('inf')

        for point in self.radar_points:
            u, v = self.project_to_2d(point)  # 레이더 포인트를 2D로 투영
            distance = np.sqrt((u - obj_center_x)**2 + (v - obj_center_y)**2)

            if distance < min_distance:
                min_distance = distance
                closest_point = point

        return closest_point

    def display_fusion_data(self, frame):
        # 카메라에서 감지된 객체와 레이더 데이터를 매칭
        if not self.detected_objects or not self.radar_points:
            # 데이터를 모두 받지 못한 경우, 그냥 원본 프레임을 표시
            cv2.imshow('Fusion Output', frame)
            cv2.waitKey(1)
            return

        for obj in self.detected_objects:
            # 바운딩 박스 그리기
            x, y, w, h = obj['bbox']
            class_name = obj['class']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 객체에 매칭되는 레이더 포인트 찾기
            radar_point = self.match_radar_to_object(obj)

            if radar_point:
                distance = np.sqrt(radar_point[0]**2 + radar_point[1]**2 + radar_point[2]**2)
                velocity = radar_point[3]

                # 객체 정보 (클래스, 거리, 속도) 표시
                label = f"{class_name}: {distance:.2f} m, {velocity:.2f} m/s"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 프레임을 OpenCV 창으로 출력
        cv2.imshow('Fusion Output', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

