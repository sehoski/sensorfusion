import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()
        self.cam_index = 4  # 기본 웹캠은 0, 외부 카메라는 1 이상의 번호
        self.cap = cv2.VideoCapture(self.cam_index)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # YOLOv8 모델 로드
        self.model = YOLO('my_camera_pkg/models/yolov8n.pt')

        # 캘리브레이션 결과 로드
        calibration_data = np.load('/home/seho/ros2_ws/src/my_camera_pkg/my_camera_pkg/calibration_result.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.new_camera_matrix = None

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]

            if self.new_camera_matrix is None:
                # 새로운 카메라 매트릭스 계산
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))

            # 이미지 왜곡 보정
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

            # 이미지 해상도 조정 (속도 최적화를 위해 해상도 낮추기)
            resized_frame = cv2.resize(undistorted_frame, (160, 120))
        
            # YOLOv8을 사용하여 객체 감지 수행
            results = self.model.predict(source=resized_frame, imgsz=160)
            annotated_frame = results[0].plot()
            
            # 원본 해상도로 프레임 다시 조정 (선택 사항)
            annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

            try:
                msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()  # 타임스탬프 추가
                self.publisher_.publish(msg)
                self.get_logger().info(f'Published camera frame with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
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

