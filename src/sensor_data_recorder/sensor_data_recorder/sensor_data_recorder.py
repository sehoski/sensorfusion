import can
import rclpy
from rclpy.node import Node
import rosbag2_py
import numpy as np
import signal
import sys
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import std_msgs.msg
from cv_bridge import CvBridge

class SensorDataRecorderNode(Node):
    def __init__(self):
        super().__init__('sensor_data_recorder_node')

        # CAN 버스 초기화
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')

        # rosbag2 writer 초기화
        self.bag_writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri='sensor_data', storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        self.bag_writer.open(storage_options, converter_options)

        # PointCloud2 메시지 타입을 rosbag에 등록
        radar_topic_info = rosbag2_py.TopicMetadata(name='/radar/points', type='sensor_msgs/msg/PointCloud2', serialization_format='cdr')
        self.bag_writer.create_topic(radar_topic_info)

        # Image 메시지 타입을 rosbag에 등록
        camera_topic_info = rosbag2_py.TopicMetadata(name='/camera/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr')
        self.bag_writer.create_topic(camera_topic_info)

        # 타이머 설정 (0.06초마다 레이더 데이터 수집)
        self.timer = self.create_timer(0.06, self.timer_callback)

        # 카메라 데이터 수신
        self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

        # cv_bridge 초기화
        self.bridge = CvBridge()

        # 타겟 상태를 관리하는 딕셔너리
        self.target_states = {}

    def timer_callback(self):
        # 레이더 데이터 수집 및 rosbag 저장
        try:
            batch_size = 10
            points = []
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    can_id = message.arbitration_id
                    raw_data = message.data
                    if 0x401 <= can_id <= 0x4FF:
                        parsed_data, _ = self.parse_can_message(raw_data)
                        if parsed_data:
                            point = self.process_radar_data(parsed_data, can_id)
                            if point:
                                points.append(point)
            
            if points:
                self.save_radar_to_bag(points)
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def camera_callback(self, msg):
        # 카메라 데이터를 rosbag에 저장
        self.bag_writer.write('/camera/image_raw', self.get_clock().now().to_msg(), msg)

    def process_radar_data(self, parsed_data, can_id):
        # 필터링 없이 원본 데이터를 기록하기 위한 함수
        if can_id not in self.target_states:
            self.target_states[can_id] = {
                'id_401_count': 0,
                'first_data': None
            }

        target_state = self.target_states[can_id]
        target_state['id_401_count'] += 1

        if target_state['id_401_count'] == 1:
            target_state['first_data'] = parsed_data
        elif target_state['id_401_count'] == 2:
            first_data = target_state['first_data']
            x = first_data['range_m'] * np.cos(np.radians(first_data['azimuth_angle']))
            y = first_data['range_m'] * np.sin(np.radians(first_data['azimuth_angle']))
            z = first_data['range_m'] * np.sin(np.radians(parsed_data['elevation']))

            target_state['id_401_count'] = 0
            target_state['first_data'] = None

            return [x, y, z, first_data['speed_radial'], parsed_data['rcs']]
        return None

    def save_radar_to_bag(self, points):
        # 레이더 데이터 PointCloud2 형식으로 생성 후 rosbag에 저장
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
        self.bag_writer.write('/radar/points', self.get_clock().now().to_msg(), pc2)

    def parse_can_message(self, data):
        if len(data) != 8:
            raise ValueError("Expected 8 bytes of data")

        swapped_data = np.frombuffer(data, dtype=np.uint8)[::-1]
        bits = ''.join(np.binary_repr(byte, width=8) for byte in swapped_data)[:64]

        parsed_data = {
            'range_m': int(bits[-14:-1], 2) * 0.04,
            'azimuth_angle': (int(bits[-32:-22], 2) - 511) * 0.16,
            'speed_radial': (int(bits[-51:-39], 2) - 2992) * 0.04,
            'rcs': int(bits[-9:-1], 2) * 0.2 - 15,
            'power': int(bits[-17:-9], 2),
            'noise': int(bits[-25:-17], 2) * 0.5,
            'elevation': (int(bits[-47:-37], 2) - 511) * 0.04
        }

        return parsed_data, bits

    def destroy_node(self):
        # CAN 버스와 rosbag 파일을 안전하게 닫음
        self.bus.shutdown()
        self.bag_writer.close()
        self.get_logger().info('CAN bus and rosbag file have been closed.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    sensor_data_recorder_node = SensorDataRecorderNode()

    def signal_handler(sig, frame):
        sensor_data_recorder_node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    rclpy.spin(sensor_data_recorder_node)

    sensor_data_recorder_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

