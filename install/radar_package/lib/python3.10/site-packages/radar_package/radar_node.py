import can
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import matplotlib.pyplot as plt
import signal
import sys

def distance_to_rgb(distance, max_distance=327.64):
    distance_min = 0
    distance_max = max_distance  # 거리의 최대값 설정 (필요에 따라 조정 가능)
    normalized_distance = (distance - distance_min) / (distance_max - distance_min)
    normalized_distance = np.clip(normalized_distance, 0, 1)
    
    # 색상 계산 (가까울수록 빨강색, 멀어질수록 파란색)
    red = int((1 - normalized_distance) * 255)
    blue = int(normalized_distance * 255)
    green = 0  # 중간색 없이 빨강에서 파랑으로 변환

    return struct.unpack('I', struct.pack('BBBB', blue, green, red, 255))[0]
    
class ExtendedKalmanFilter:
    def __init__(self, f, h, F, H, Q, R, P, x):
        self.f = f
        self.h = h
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self, u=0):
        self.x = self.f(self.x, u)
        self.P = np.dot(np.dot(self.F(self.x, u), self.P), self.F(self.x, u).T) + self.Q

    def update(self, z):
        y = z - self.h(self.x)
        S = np.dot(self.H(self.x), np.dot(self.P, self.H(self.x).T)) + self.R
        K = np.dot(np.dot(self.P, self.H(self.x).T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = (I - np.dot(K, self.H(self.x))).dot(self.P)

def f(x, u):
    dt = 0.05  # 타이머 주기와 일치하도록 설정
    return np.array([
        x[0] + x[3] * dt * np.cos(x[2]),  # x 위치
        x[1] + x[3] * dt * np.sin(x[2]),  # y 위치
        x[2],  # 방향 (여기서는 각속도가 없으므로 그대로 유지)
        x[3]  # 속도
    ])

def h(x):
    return np.array([x[0], x[1], x[2], x[3]])  # 모든 상태를 관측 가능하다고 가정

def F(x, u):
    dt = 0.05
    return np.array([
        [1, 0, -x[3] * dt * np.sin(x[2]), dt * np.cos(x[2])],
        [0, 1, x[3] * dt * np.cos(x[2]), dt * np.sin(x[2])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def H(x):
    return np.eye(4)  # 관측 행렬은 단위 행렬로 가정

class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')
        self.publisher_ = self.create_publisher(PointCloud2, 'radar/points', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.target_data = []
        self.raw_data = []  # 필터링되지 않은 데이터를 저장하기 위한 리스트

        # EKF 초기화
        Q = np.eye(4) * 0.01
        R = np.eye(4) * 0.1
        P = np.eye(4)
        x = np.zeros(4)
        self.ekf = ExtendedKalmanFilter(f, h, F, H, Q, R, P, x)

    def timer_callback(self):
        try:
            batch_size = 10
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    can_id = message.arbitration_id
                    if 0x401 <= can_id <= 0x4FF:
                        parsed_data = self.parse_can_message(message.data)
                        if parsed_data:
                            self.process_radar_data(parsed_data)
            if self.target_data:
                self.publish_points(self.target_data)
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")
        self.send_transform()

    def process_radar_data(self, parsed_data):
        distance = parsed_data['range_m']
        azimuth_angle = parsed_data['azimuth_angle']
        speed = parsed_data['speed_radial']
        elevation = parsed_data['elevation']
        noise = parsed_data['noise']

        if distance > 327.64:  # 거리가 10m 이상인 데이터를 무시
            return

        if noise > 30:  # 노이즈 수준이 높은 데이터를 무시
            return

        x = distance * np.cos(np.radians(azimuth_angle)) * np.cos(np.radians(elevation))
        y = distance * np.sin(np.radians(azimuth_angle)) * np.cos(np.radians(elevation))
        z = distance * np.sin(np.radians(elevation))
        rgb = distance_to_rgb(distance)

        # 필터링되지 않은 데이터를 저장
        self.raw_data.append((x, y, z))

        # EKF 업데이트
        observation = np.array([x, y, azimuth_angle, speed])
        self.ekf.predict(u=0)
        self.ekf.update(observation)

        # 필터링된 상태 추정치 사용
        filtered_x, filtered_y, filtered_azimuth, filtered_speed = self.ekf.x
        self.target_data.append((filtered_x, filtered_y, z, rgb))

    def parse_can_message(self, data):
        if len(data) != 8:
            raise ValueError("Expected 8 bytes of data")

        swapped_data = bytearray(8)
        for i in range(8):
            swapped_data[i] = data[7 - i]

        bits = ''.join(f'{byte:08b}' for byte in swapped_data)

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

        return {
            'range_m': range_m,
            'azimuth_angle': azimuth_angle,
            'speed_radial': speed_radial,
            'rcs': rcs,
            'power': power,
            'noise': noise,
            'elevation': elevation
        }

    def publish_points(self, points):
        if not points:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'radar_frame'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        point_cloud = pc2.create_cloud(header, fields, points)
        self.publisher_.publish(point_cloud)

    def send_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'radar_frame'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    radar_node = RadarNode()

    def signal_handler(sig, frame):
        radar_node.destroy_node()
        rclpy.shutdown()
        plot_data(radar_node.raw_data, radar_node.target_data)
        raise SystemExit

    signal.signal(signal.SIGINT, signal_handler)

    rclpy.spin(radar_node)

def plot_data(raw_data, filtered_data):
    if raw_data and filtered_data:
        raw_x, raw_y, raw_z = zip(*raw_data)
        filtered_x, filtered_y, filtered_z, _ = zip(*filtered_data)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(raw_x, raw_y, c='blue', label='Raw Data')
        plt.title('Raw Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(filtered_x, filtered_y, c='red', label='Filtered Data')
        plt.title('Filtered Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.show()
    else:
        print("No data to plot.")

if __name__ == '__main__':
    main()

