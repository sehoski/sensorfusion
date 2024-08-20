import can
import rclpy
from rclpy.node import Node
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt

class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.timer = self.create_timer(0.06, self.timer_callback)
        self.target_states = {}  # Target 상태를 관리하는 딕셔너리
        self.filtered_data = []  # 필터링된 데이터를 저장할 리스트

        # 신뢰할 수 있는 노이즈 값의 임계 범위 설정 (30 dB ~ 90 dB)
        self.noise_threshold = (30, 90)

    def timer_callback(self):
        try:
            batch_size = 10  # 한 번에 처리할 메시지 개수
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    can_id = message.arbitration_id
                    raw_data = message.data.hex()  # 8바이트 데이터를 16진수 문자열로 변환
                    if 0x401 <= can_id <= 0x4FF:  # ID가 0x401에서 0x4FF 사이인 경우만 처리
                        parsed_data, bit_info = self.parse_can_message(message.data)
                        if parsed_data:
                            # 노이즈 임계값을 초과하는 데이터는 무시
                            if self.check_noise_threshold(parsed_data['noise']):
                                self.process_radar_data(parsed_data, can_id, raw_data)
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def check_noise_threshold(self, noise):
        min_noise, max_noise = self.noise_threshold
        if not (min_noise <= noise <= max_noise):
            self.get_logger().warn(f"Noise value {noise} dB out of bounds, ignoring.")
            return False
        return True

    def process_radar_data(self, parsed_data, can_id, raw_data):
        # 각 타깃의 상태를 추적하기 위한 초기화
        if can_id not in self.target_states:
            self.target_states[can_id] = {
                'id_401_count': 0,
                'first_data': None,
                'ekf': self.initialize_ekf()  # 확장된 칼만 필터 초기화
            }

        target_state = self.target_states[can_id]
        target_state['id_401_count'] += 1

        if target_state['id_401_count'] == 1:
            # 첫 번째 데이터 처리
            swapped_data_1 = self.swap_and_convert_to_64bit(raw_data)
            target_state['first_data'] = {
                'x': parsed_data['x'],
                'y': parsed_data['y'],
                'z': parsed_data['z'],
                'speed_radial': parsed_data['speed_radial'],
                'raw_data_1': raw_data,
                'swapped_data_1': swapped_data_1['swapped_data'],
                'swapped_data_1_bin': swapped_data_1['swapped_data_bin']
            }
        elif target_state['id_401_count'] == 2:
            # 두 번째 데이터 처리
            swapped_data_2 = self.swap_and_convert_to_64bit(raw_data)
            ekf = target_state['ekf']

            # 확장된 칼만 필터 업데이트
            z = np.array([parsed_data['x'], parsed_data['y'], parsed_data['z'], parsed_data['speed_radial']])
            ekf.predict()
            ekf.update(z)

            # 필터링된 결과 저장
            filtered_state = ekf.x
            self.filtered_data.append({
                'can_id': can_id,
                'x': filtered_state[0],
                'y': filtered_state[1],
                'z': filtered_state[2],
                'speed_radial': filtered_state[3]
            })

            # 타깃 상태 초기화
            target_state['id_401_count'] = 0
            target_state['first_data'] = None

    def initialize_ekf(self):
        # 초기 상태 벡터 (x, y, z, speed)
        x = np.zeros(4)

        # 초기 오차 공분산 행렬
        P = np.diag([1, 1, 1, 1])

        # 상태 전이 행렬
        F = np.eye(4)

        # 측정 행렬
        H = np.eye(4)

        # 측정 오차 공분산 행렬
        R = np.diag([0.1, 0.1, 0.1, 0.1])

        # 프로세스 노이즈 공분산 행렬
        Q = np.diag([0.1, 0.1, 0.1, 0.1])

        return ExtendedKalmanFilter(x, P, F, H, R, Q)

    def swap_and_convert_to_64bit(self, raw_data):
        # 16진수 문자열을 바이트 배열로 변환
        byte_array = bytearray.fromhex(raw_data)
        # 바이트 순서 반전
        swapped_byte_array = byte_array[::-1]
        # 64비트 이진수로 변환
        swapped_data_bin = ''.join(f"{byte:08b}" for byte in swapped_byte_array)[:64]
        # 바이트 배열을 다시 16진수 문자열로 변환
        swapped_data_hex = swapped_byte_array.hex()

        return {
            'swapped_data': swapped_data_hex,
            'swapped_data_bin': swapped_data_bin
        }

    def parse_can_message(self, data):
        if len(data) != 8:
            raise ValueError("Expected 8 bytes of data")

        # Swap the data bytes and convert to 64-bit binary
        swapped_data = np.frombuffer(data, dtype=np.uint8)[::-1]
        bits = ''.join(np.binary_repr(byte, width=8) for byte in swapped_data)[:64]

        # 데이터 파싱 (range, azimuth, elevation, speed 등으로 가정)
        range_bits = bits[-14:-1]
        range_raw = int(range_bits, 2)
        range_m = range_raw * 0.04

        azimuth_bits = bits[-32:-22]
        azimuth_raw = int(azimuth_bits, 2)
        azimuth_angle = (azimuth_raw - 511) * 0.16

        elevation_bits = bits[-51:-39]
        elevation_raw = int(elevation_bits, 2)
        elevation = (elevation_raw - 511) * 0.04

        speed_bits = bits[-9:-1]
        speed_raw = int(speed_bits, 2)
        speed_radial = speed_raw * 0.2 - 15

        # 변환 공식을 이용하여 x, y, z 좌표 계산
        x = range_m * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth_angle))
        y = range_m * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth_angle))
        z = range_m * np.sin(np.radians(elevation))

        parsed_data = {
            'x': x,
            'y': y,
            'z': z,
            'speed_radial': speed_radial,
            'noise': noise
        }

        return parsed_data, bits

    def destroy_node(self):
        self.bus.shutdown()
        self.visualize_data()
        self.get_logger().info('CAN bus has been shut down properly.')
        super().destroy_node()

    def visualize_data(self):
        xs = [data['x'] for data in self.filtered_data]
        ys = [data['y'] for data in self.filtered_data]
        zs = [data['z'] for data in self.filtered_data]
        speeds = [data['speed_radial'] for data in self.filtered_data]

        plt.figure(figsize=(12, 8))

        plt.subplot(4, 1, 1)
        plt.plot(xs, label='X Position (m)')
        plt.title('Filtered X Position')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(ys, label='Y Position (m)')
        plt.title('Filtered Y Position')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(zs, label='Z Position (m)')
        plt.title('Filtered Z Position')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(speeds, label='Speed Radial (m/s)')
        plt.title('Filtered Speed Radial')
        plt.legend()

        plt.tight_layout()
        plt.show()

class ExtendedKalmanFilter:
    def __init__(self, x, P, F, H, R, Q):
        self.x = x
        self.P = P
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q

    def predict(self):
        # 예측 단계
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # 측정 업데이트 단계
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

def main(args=None):
    rclpy.init(args=args)
    radar_node = RadarNode()

    def signal_handler(sig, frame):
        radar_node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    rclpy.spin(radar_node)

if __name__ == '__main__':
    main()

