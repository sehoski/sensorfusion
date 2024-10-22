import can
import rclpy
from rclpy.node import Node
import numpy as np
import signal
import sys
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import std_msgs.msg
import json
import os
from datetime import datetime
import cv2
from cv_bridge import CvBridge

def create_unique_directory(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(base_dir, f"sensor_data_{timestamp}")
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(base_dir, f"sensor_data_{timestamp}_{counter}")
        counter += 1
    os.makedirs(save_path)
    return save_path

class RadarCameraNode(Node):
    def __init__(self):
        super().__init__('radar_camera_node')
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bridge = CvBridge()
        self.target_states = {}  # 타겟 상태를 관리하는 딕셔너리

        # 퍼블리셔 생성
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/radar/points', 10)
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        # 카메라 설정 및 실제 적용된 설정 확인
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도: 너비 640
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # 해상도: 높이 480
        self.cap.set(cv2.CAP_PROP_FPS, 30)             # 프레임 속도: 20fps

        # 설정이 제대로 적용되었는지 확인
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 설정 값과 실제 값 출력
        self.get_logger().info(f"Requested resolution: 1280x800 @ 30fps")
        self.get_logger().info(f"Actual resolution: {actual_width}x{actual_height} @ {actual_fps}fps")

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            raise RuntimeError("Failed to open camera")

        # 실제 적용된 해상도와 FPS로 VideoWriter 초기화
        base_dir = os.path.expanduser('~/ros2_ws/src/sensor_data')
        self.save_path = create_unique_directory(base_dir)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = os.path.join(self.save_path, 'camera_recording.avi')
        self.out = cv2.VideoWriter(video_filename, self.fourcc, actual_fps, (int(actual_width), int(actual_height)))

        # 데이터 저장 설정
        self.radar_file = open(os.path.join(self.save_path, 'radar_data.json'), 'w')

        # 타이머 설정
        self.create_timer(1 / 40, self.radar_callback)  # 레이더 데이터 처리 (40Hz)
        self.create_timer(1 / 30, self.camera_callback)  # 카메라 데이터 처리 (30Hz)

    def radar_callback(self):
        try:
            batch_size = 20  # 한 번에 처리할 메시지 개수
            points = []
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    can_id = message.arbitration_id
                    raw_data = message.data.hex()  # 8바이트 데이터를 16진수 문자열로 변환
                    if 0x401 <= can_id <= 0x4FF:  # ID가 0x401에서 0x4FF 사이인 경우만 처리
                        parsed_data, bit_info = self.parse_can_message(message.data)
                        if parsed_data:
                            point = self.process_radar_data(parsed_data, can_id, raw_data)
                            if point:
                                points.append(point)
                                self.save_radar_data(can_id, raw_data, parsed_data, point)
            
            if points:
                self.publish_point_cloud(points)
        except Exception as e:
            self.get_logger().error(f"Error in radar_callback: {e}")

    def camera_callback(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                # 이미지를 ROS 메시지로 변환하고 발행
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_pub.publish(img_msg)

                # 비디오 녹화
                if self.out is None:
                    height, width, _ = frame.shape
                    video_filename = os.path.join(self.save_path, 'camera_recording.avi')
                    self.out = cv2.VideoWriter(video_filename, self.fourcc, 30.0, (width, height))

                self.out.write(frame)

            except Exception as e:
                self.get_logger().error(f"Error in camera_callback: {e}")

    def process_radar_data(self, parsed_data, can_id, raw_data):
        if can_id not in self.target_states:
            self.target_states[can_id] = {
                'id_401_count': 0,
                'first_data': None
            }

        target_state = self.target_states[can_id]
        target_state['id_401_count'] += 1

        if target_state['id_401_count'] == 1:
            target_state['first_data'] = parsed_data
            self.get_logger().debug(f"First data received for CAN ID {can_id}:")
            self.get_logger().debug(f"  Distance: {parsed_data['range_m']:.2f} m")
            self.get_logger().debug(f"  Speed Radial: {parsed_data['speed_radial']:.2f} m/s")
            self.get_logger().debug(f"  Azimuth_angle: {parsed_data['azimuth_angle']:.2f} degrees")
        elif target_state['id_401_count'] == 2:
            first_data = target_state['first_data']
            self.get_logger().debug(f"Second data received for CAN ID {can_id}:")
            self.get_logger().debug(f"  RCS: {parsed_data['rcs']:.2f} dB")
            self.get_logger().debug(f"  Power: {parsed_data['power']:.2f} dB")
            self.get_logger().debug(f"  Noise: {parsed_data['noise']:.2f} dB")
            self.get_logger().debug(f"  Elevation: {parsed_data['elevation']:.2f} degrees")

            # 극좌표계를 직교좌표계로 변환
            x = first_data['range_m'] * np.cos(np.radians(first_data['azimuth_angle']))
            y = first_data['range_m'] * np.sin(np.radians(first_data['azimuth_angle']))
            z = first_data['range_m'] * np.sin(np.radians(parsed_data['elevation']))

            self.get_logger().info(f"Converted coordinates: x={x}, y={y}, z={z}")
            self.get_logger().info(f"Published point: x={x}, y={y}, z={z}")

            target_state['id_401_count'] = 0
            target_state['first_data'] = None

            return [x, y, z, first_data['speed_radial'], parsed_data['rcs']]

        return None

    def publish_point_cloud(self, points):
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "radar_frame"

        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='velocity', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='intensity', offset=16, datatype=point_cloud2.PointField.FLOAT32, count=1),
        ]

        pc2 = point_cloud2.create_cloud(header, fields, points)
        self.point_cloud_pub.publish(pc2)

    def save_radar_data(self, can_id, raw_data, parsed_data, point):
        data = {
            "timestamp": self.get_clock().now().to_msg().sec,
            "can_id": can_id,
            "raw_data": raw_data,
            "parsed_data": parsed_data,
            "point": point
        }
        json.dump(data, self.radar_file)
        self.radar_file.write('\n')
        self.radar_file.flush()

    def parse_can_message(self, data):
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

        rcs_bits = bits[-9:-1]
        rcs_raw = int(rcs_bits, 2)
        rcs = rcs_raw * 0.2 - 15

        power_bits = bits[-17:-9]
        power_raw = int(power_bits, 2)
        power = power_raw

        noise_bits = bits[-25:-17]
        noise_raw = int(noise_bits, 2)
        noise = noise_raw * 0.5

        elevation_bits = bits[-47:-37]
        elevation_raw = int(elevation_bits, 2)
        elevation = (elevation_raw - 511) * 0.04

        parsed_data = {
            'range_m': range_m,
            'azimuth_angle': azimuth_angle,
            'speed_radial': speed_radial,
            'rcs': rcs,
            'power': power,
            'noise': noise,
            'elevation': elevation
        }

        bit_info = {
            'reversed_64bit': bits
        }

        return parsed_data, bit_info

    def destroy_node(self):
        if hasattr(self, 'radar_file'):
            self.radar_file.close()
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
        self.bus.shutdown()
        self.get_logger().info('CAN bus, camera, and data files have been shut down properly.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RadarCameraNode()

    def signal_handler(sig, frame):
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

