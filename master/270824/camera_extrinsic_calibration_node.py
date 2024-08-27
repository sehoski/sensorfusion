import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.reflector_publisher = self.create_publisher(PointStamped, 'camera/reflector_position', 10)
        self.bridge = CvBridge()

        calibration_data = np.load('/home/seho/ros2_ws/src/my_camera_pkg/my_camera_pkg/calibration_result.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']

        self.cap = cv2.VideoCapture(4)
        self.new_camera_matrix = None

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
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image from camera')
            return

        if self.new_camera_matrix is None:
            h, w = frame.shape[:2]
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )

        undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

        ros_image = self.bridge.cv2_to_imgmsg(undistorted_frame, encoding="bgr8")
        self.image_publisher.publish(ros_image)

        self.get_logger().info('Published camera image')

        center, contour = self.detect_reflector(undistorted_frame)

        if center is not None:
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "camera_frame"
            point_msg.point.x = float(center[0])
            point_msg.point.y = float(center[1])
            point_msg.point.z = 0.0

            self.reflector_publisher.publish(point_msg)
            self.get_logger().info(f"Published reflector position: {center}")

            # 3D 점 업데이트
            x = (center[0] - undistorted_frame.shape[1]/2) / undistorted_frame.shape[1]
            y = -(center[1] - undistorted_frame.shape[0]/2) / undistorted_frame.shape[0]
            z = 0  # 고정 거리 (실제 거리를 알 수 없으므로 임의의 값 사용)

            self.scatter._offsets3d = ([x], [y], [z])

            plt.draw()
            plt.pause(0.001)

            cv2.circle(undistorted_frame, center, 5, (0, 255, 0), -1)
            cv2.drawContours(undistorted_frame, [contour], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(undistorted_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(undistorted_frame, f"Reflector at: {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Reflector Detection', undistorted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            plt.close(self.fig)
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()

    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
