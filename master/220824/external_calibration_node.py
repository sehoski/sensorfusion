import rclpy
from rclpy.node import Node
import can
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import json
import os

class ExternalCalibrationNode(Node):
    def __init__(self):
        super().__init__('external_calibration_node')
        
        # 레이더 설정
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.radar_timer = self.create_timer(0.06, self.radar_callback)
        self.collected_radar_data = []
        self.target_states = {}
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(4)
        self.camera_timer = self.create_timer(0.1, self.camera_callback)
        self.detected_positions = []
        
        # 캘리브레이션 데이터 로드
        calibration_data = np.load('/home/seho/ros2_ws/src/my_camera_pkg/my_camera_pkg/calibration_result.npz')
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.new_camera_matrix = None
        
        # 시각화 설정
        self.fig = plt.figure(figsize=(18, 6))
        self.ax_radar = self.fig.add_subplot(131, projection='3d')
        self.ax_camera = self.fig.add_subplot(132)
        self.ax_combined = self.fig.add_subplot(133)
        
        plt.ion()
        plt.show()
        
        # 외부 캘리브레이션 변수
        self.transformation_matrix = None
        self.calibration_points_radar = []
        self.calibration_points_camera = []
        
        # 캘리브레이션 상태
        self.is_calibrated = False
        self.is_calibrating = False
        
        # 캘리브레이션 파일 경로
        self.calibration_directory = '/home/seho/ros2_ws/master/'
        self.calibration_file = os.path.join(self.calibration_directory, 'external_calibration.json')

        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(self.calibration_directory):
            os.makedirs(self.calibration_directory)
            print(f"Created directory: {self.calibration_directory}")
        
        # 저장된 캘리브레이션 데이터 로드
        self.load_calibration()

        # 현재 레이더 및 카메라 데이터
        self.current_radar_data = None
        self.current_camera_data = None

    def radar_callback(self):
        try:
            message = self.bus.recv(0.005)
            if message is not None:
                can_id = message.arbitration_id
                if 0x401 <= can_id <= 0x4FF:
                    parsed_data = self.parse_radar_message(message.data)
                    if parsed_data:
                        self.process_radar_data(parsed_data, can_id)
        except Exception as e:
            self.get_logger().error(f"Error in radar_callback: {e}")

    def camera_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        if self.new_camera_matrix is None:
            h, w = frame.shape[:2]
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        
        undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        
        center, contour = self.detect_reflector(undistorted_frame)
        
        if center is not None:
            self.detected_positions.append(center)
            cv2.circle(undistorted_frame, center, 5, (0, 255, 0), -1)
            cv2.drawContours(undistorted_frame, [contour], -1, (0, 255, 0), 2)
            
            self.current_camera_data = center
        
        self.update_visualization(undistorted_frame)

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

    def parse_radar_message(self, data):
        if len(data) != 8:
            raise ValueError("Expected 8 bytes of data")

        swapped_data = np.frombuffer(data, dtype=np.uint8)[::-1]
        bits = ''.join(np.binary_repr(byte, width=8) for byte in swapped_data)[:64]

        range_bits = bits[-14:-1]
        range_raw = int(range_bits, 2)
        range_m = range_raw * 0.04

        azimuth_bits = bits[-32:-22]
        azimuth_raw = int(azimuth_bits, 2)
        azimuth_angle = (azimuth_raw - 511) * 0.16

        speed_bits = bits[-51:-39]
        speed_raw = int(speed_bits, 2)
        speed_radial = (speed_raw - 2992) * 0.04

        elevation_bits = bits[-47:-37]
        elevation_raw = int(elevation_bits, 2)
        elevation = (elevation_raw - 511) * 0.04

        parsed_data = {
            'range_m': range_m,
            'azimuth_angle': azimuth_angle,
            'speed_radial': speed_radial,
            'elevation': elevation
        }

        return parsed_data

    def process_radar_data(self, parsed_data, can_id):
        x = parsed_data['range_m'] * np.cos(np.radians(parsed_data['elevation'])) * np.cos(np.radians(parsed_data['azimuth_angle']))
        y = parsed_data['range_m'] * np.cos(np.radians(parsed_data['elevation'])) * np.sin(np.radians(parsed_data['azimuth_angle']))
        z = parsed_data['range_m'] * np.sin(np.radians(parsed_data['elevation']))

        new_data = {
            'can_id': can_id,
            'x': x,
            'y': y,
            'z': z,
            'speed_radial': parsed_data['speed_radial']
        }

        self.collected_radar_data.append(new_data)
        self.current_radar_data = (x, y, z)

    def update_visualization(self, camera_frame):
        # 레이더 데이터 시각화
        self.ax_radar.clear()
        if self.collected_radar_data:
            x_data = [data['x'] for data in self.collected_radar_data]
            y_data = [data['y'] for data in self.collected_radar_data]
            z_data = [data['z'] for data in self.collected_radar_data]
            self.ax_radar.scatter(x_data, y_data, z_data, c='blue', marker='o')
        self.ax_radar.set_xlabel('X (m)')
        self.ax_radar.set_ylabel('Y (m)')
        self.ax_radar.set_zlabel('Z (m)')
        self.ax_radar.set_title('Radar Data')

        # 카메라 데이터 시각화
        self.ax_camera.clear()
        self.ax_camera.imshow(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
        self.ax_camera.set_title('Camera Data')

        # 결합된 데이터 시각화
        self.ax_combined.clear()
        self.ax_combined.imshow(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
        if self.is_calibrated and self.transformation_matrix is not None:
            for radar_point in self.collected_radar_data:
                radar_coords = np.array([radar_point['x'], radar_point['y'], radar_point['z'], 1])
                camera_coords = self.transformation_matrix.dot(radar_coords)
                x, y = camera_coords[:2] / camera_coords[2]
                self.ax_combined.plot(x, y, 'ro', markersize=5)
        self.ax_combined.set_title('Combined Data')

        plt.draw()
        plt.pause(0.01)

    def calculate_transformation_matrix(self):
        if len(self.calibration_points_radar) < 4 or len(self.calibration_points_camera) < 4:
            self.get_logger().warn("Not enough calibration points")
            return False

        # 최소 제곱법을 사용하여 변환 행렬 계산
        radar_points = np.array(self.calibration_points_radar)
        camera_points = np.array(self.calibration_points_camera)

        # 호모그래피 행렬 계산 (이것은 근사값입니다)
        H, _ = cv2.findHomography(radar_points[:, :2], camera_points)

        # 완전한 3D to 2D 변환 행렬 생성
        self.transformation_matrix = np.vstack([H, [0, 0, 1]])
        
        return True

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_points_radar = []
        self.calibration_points_camera = []
        self.get_logger().info("Calibration started. Move the device to different positions and use 'p' to capture points.")

    def stop_calibration(self):
        self.is_calibrating = False
        success = self.calculate_transformation_matrix()
        if success:
            self.is_calibrated = True
            self.save_calibration()
            self.get_logger().info("Calibration completed and saved.")
        else:
            self.get_logger().warn("Calibration failed. Please try again.")

    def capture_calibration_point(self):
        if self.is_calibrating:
            if self.current_radar_data is not None and self.current_camera_data is not None:
                self.calibration_points_radar.append(self.current_radar_data)
                self.calibration_points_camera.append(self.current_camera_data)
                print(f"Calibration point captured. Total points: {len(self.calibration_points_radar)}")
            else:
                print("No data available for calibration point. Please try again.")
        else:
            print("Calibration not started. Use 'c' to start calibration.")

    def save_calibration(self):
        calibration_data = {
            'transformation_matrix': self.transformation_matrix.tolist()
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f)

    def load_calibration(self):
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                calibration_data = json.load(f)
            self.transformation_matrix = np.array(calibration_data['transformation_matrix'])
            self.is_calibrated = True
            self.get_logger().info("Loaded saved calibration data.")
        else:
            self.get_logger().info("No saved calibration data found.")

    def run(self):
        print("External Calibration Node is running.")
        print("Commands:")
        print("  'c': Start calibration")
        print("  's': Stop calibration")
        print("  'p': Capture calibration point")
        print("  'q': Quit")

        while rclpy.ok():
            rclpy.spin_once(self)
            
            user_input = input("Enter command: ").lower()
            if user_input == 'c':
                self.start_calibration()
            elif user_input == 's':
                self.stop_calibration()
            elif user_input == 'p':
                self.capture_calibration_point()
            elif user_input == 'q':
                break

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
