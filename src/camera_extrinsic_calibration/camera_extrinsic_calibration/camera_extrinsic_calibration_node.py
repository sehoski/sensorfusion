import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StereoCameraNode(Node):
    def __init__(self):
        super().__init__('stereo_camera_node')

        self.left_image_publisher = self.create_publisher(Image, 'camera/left/image_raw', 10)
        self.right_image_publisher = self.create_publisher(Image, 'camera/right/image_raw', 10)
        self.reflector_publisher = self.create_publisher(PointStamped, 'camera/reflector_position', 10)
        self.bridge = CvBridge()

        # 스테레오 카메라 캘리브레이션 데이터 로드
        calib_data = np.load('/path/to/stereo_calibration.npz')
        self.camera_matrix_left = calib_data['camera_matrix_left']
        self.dist_coeffs_left = calib_data['dist_coeffs_left']
        self.camera_matrix_right = calib_data['camera_matrix_right']
        self.dist_coeffs_right = calib_data['dist_coeffs_right']
        self.R = calib_data['R']  # 회전 행렬
        self.T = calib_data['T']  # 변환 벡터

        # 스테레오 정류화 맵 계산
        self.stereo_rectify()

        # 스테레오 카메라 설정
        self.cap_left = cv2.VideoCapture(4)  # 왼쪽 카메라
        self.cap_right = cv2.VideoCapture(5)  # 오른쪽 카메라

        # Matplotlib 초기화
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.scatter = self.ax.scatter([], [], [], c='r', marker='o')

        self.timer = self.create_timer(0.1, self.capture_and_process_image)

    def stereo_rectify(self):
        h, w = 480, 640  # 카메라 해상도에 맞게 조정
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            (w, h), self.R, self.T
        )
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
        )
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
        )
        self.Q = Q  # 시차를 깊이로 변환하는 데 사용되는 행렬

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
                return (cx, cy), max_contour

        return None, None

    def capture_and_process_image(self):
        ret_left, frame_left = self.cap_left.read()
        ret_right, frame_right = self.cap_right.read()

        if not ret_left or not ret_right:
            self.get_logger().error('Failed to capture image from stereo camera')
            return

        # 스테레오 정류화 적용
        frame_left_rect = cv2.remap(frame_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        frame_right_rect = cv2.remap(frame_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        # ROS 이미지 메시지로 변환 후 퍼블리시
        left_ros_image = self.bridge.cv2_to_imgmsg(frame_left_rect, encoding="bgr8")
        right_ros_image = self.bridge.cv2_to_imgmsg(frame_right_rect, encoding="bgr8")
        self.left_image_publisher.publish(left_ros_image)
        self.right_image_publisher.publish(right_ros_image)

        self.get_logger().info('Published stereo camera images')

        # 왼쪽 이미지에서 반사체 감지
        center_left, contour_left = self.detect_reflector(frame_left_rect)

        if center_left is not None:
            # 오른쪽 이미지에서도 반사체 감지
            center_right, _ = self.detect_reflector(frame_right_rect)

            if center_right is not None:
                # 시차 계산
                disparity = center_left[0] - center_right[0]

                # 3D 좌표 계산
                point_3d = cv2.perspectiveTransform(np.array([[center_left + (disparity,)]]), self.Q)[0][0]
                x, y, z = point_3d

                # PointStamped 메시지 생성 및 퍼블리시
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "camera_frame"
                point_msg.point.x = x
                point_msg.point.y = y
                point_msg.point.z = z

                self.reflector_publisher.publish(point_msg)
                self.get_logger().info(f"Published reflector 3D position: ({x}, {y}, {z})")

                # Matplotlib 3D 점 업데이트
                self.scatter._offsets3d = ([x], [y], [z])

                plt.draw()
                plt.pause(0.001)

            # 반사체 시각화 (왼쪽 이미지에만)
            cv2.circle(frame_left_rect, center_left, 5, (0, 255, 0), -1)
            cv2.drawContours(frame_left_rect, [contour_left], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(contour_left)
            cv2.rectangle(frame_left_rect, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame_left_rect, f"Reflector at: {center_left}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 결과 화면에 표시
        cv2.imshow('Left Camera', frame_left_rect)
        cv2.imshow('Right Camera', frame_right_rect)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap_left.release()
            self.cap_right.release()
            cv2.destroyAllWindows()
            plt.close(self.fig)
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    stereo_camera_node = StereoCameraNode()

    try:
        rclpy.spin(stereo_camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        stereo_camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
