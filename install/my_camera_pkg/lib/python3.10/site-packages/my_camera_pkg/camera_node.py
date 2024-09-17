import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
        self.bridge = CvBridge()
        self.cam_index = 4  # 기본 웹캠은 0, 외부 카메라는 1 이상의 번호
        self.cap = cv2.VideoCapture(self.cam_index)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 캘리브레이션 결과 로드
        calibration_data = np.load('/home/seho/ros2_ws/src/my_camera_pkg/my_camera_pkg/calibration_result.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.new_camera_matrix = None

        # 카메라 정보 메시지 초기화
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = "camera_frame"
        self.camera_info_msg.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_info_msg.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_info_msg.distortion_model = "plumb_bob"
        self.camera_info_msg.d = self.dist_coeffs.flatten().tolist()
        self.camera_info_msg.k = self.camera_matrix.flatten().tolist()

        # YOLOv8 모델 로드
        self.model = YOLO('my_camera_pkg/models/yolov8n.pt')  # YOLOv8 모델 경로 설정

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            if self.new_camera_matrix is None:
                # 새로운 카메라 매트릭스 계산
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
                )
                self.camera_info_msg.p = self.new_camera_matrix.flatten().tolist() + [0, 0, 0]

            # 이미지 왜곡 보정
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

            # YOLOv8을 사용한 객체 감지 수행
            results = self.model.predict(source=undistorted_frame, imgsz=640)
            
            # 원하는 클래스만 필터링 (사람: 0, 자전거: 1, 자동차: 2)
            desired_classes = [0, 1, 2]
            filtered_boxes = []
            
            # YOLOv8의 boxes에서 클래스 필터링
            for result in results:
                for box in result.boxes:
                    if int(box.cls.cpu().numpy()) in desired_classes:
                        filtered_boxes.append(box)

            # 필터링된 결과로 수동으로 주석 추가
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # 바운딩 박스 좌표
                cls = int(box.cls.cpu().numpy())  # 클래스
                conf = float(box.conf.cpu().numpy().item())  # 신뢰도를 스칼라 값으로 변환
                label = f'{self.model.names[cls]} {conf:.2f}'  # 클래스 이름과 신뢰도 표시

                # 바운딩 박스 그리기
                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 클래스 이름과 신뢰도 텍스트 추가
                cv2.putText(undistorted_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            try:
                # OpenCV로 실시간 이미지 보기
                cv2.imshow('YOLOv8 Detection', undistorted_frame)
                cv2.waitKey(1)  # 적절한 프레임 속도를 유지하기 위해 필요

                # CvBridge를 사용하여 OpenCV 이미지를 ROS 메시지로 변환
                img_msg = self.bridge.cv2_to_imgmsg(undistorted_frame, 'bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "camera_frame"
                self.image_publisher.publish(img_msg)

                # 카메라 정보 퍼블리시
                self.camera_info_msg.header.stamp = img_msg.header.stamp
                self.camera_info_publisher.publish(self.camera_info_msg)

                self.get_logger().info(f'Published camera frame and info with timestamp: {img_msg.header.stamp.sec}.{img_msg.header.stamp.nanosec}')
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()  # OpenCV 창 닫기
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

