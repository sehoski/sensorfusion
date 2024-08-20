import can
import rclpy
from rclpy.node import Node
import numpy as np
import signal
import sys
import csv
from datetime import datetime

class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.timer = self.create_timer(0.06, self.timer_callback)
        self.collected_data = []
        self.target_states = {}  # 타겟 상태를 관리하는 딕셔너리

    def timer_callback(self):
        try:
            batch_size = 10  # 한 번에 처리할 메세지 개수
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    can_id = message.arbitration_id
                    raw_data = message.data.hex()  # 8바이트 데이터를 16진수 문자열로 변환
                    if 0x401 <= can_id <= 0x4FF:  # ID가 0x401에서 0x4FF 사이인 경우만 처리
                        parsed_data, bit_info = self.parse_can_message(message.data)
                        if parsed_data:
                            self.process_radar_data(parsed_data, can_id, raw_data)
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def process_radar_data(self, parsed_data, can_id, raw_data):
        # 각 타겟의 상태를 추적하기 위한 초기화
        if can_id not in self.target_states:
            self.target_states[can_id] = {
                'id_401_count': 0,
                'first_data': None
            }

        target_state = self.target_states[can_id]
        target_state['id_401_count'] += 1

        if target_state['id_401_count'] == 1:
            # 첫 번째 데이터 처리
            swapped_data_1 = self.swap_and_convert_to_64bit(raw_data)
            target_state['first_data'] = {
                'range_m': parsed_data['range_m'],
                'azimuth_angle': parsed_data['azimuth_angle'],
                'speed_radial': parsed_data['speed_radial'],
                'raw_data_1': raw_data,
                'swapped_data_1': swapped_data_1['swapped_data'],
                'swapped_data_1_bin': swapped_data_1['swapped_data_bin']
            }
            print(f"First data received for CAN ID {can_id}:")
            print(f"  Distance: {parsed_data['range_m']:.2f} m")
            print(f"  Speed Radial: {parsed_data['speed_radial']:.2f} m/s")
            print(f"  Azimuth_angle: {parsed_data['azimuth_angle']:.2f} degrees")
        elif target_state['id_401_count'] == 2:
            # 두 번째 데이터 처리
            swapped_data_2 = self.swap_and_convert_to_64bit(raw_data)
            first_data = target_state['first_data']
            print(f"Second data received for CAN ID {can_id}:")
            print(f"  RCS: {parsed_data['rcs']:.2f} dB")
            print(f"  Power: {parsed_data['power']:.2f} dB")
            print(f"  Noise: {parsed_data['noise']:.2f} dB")
            print(f"  Elevation: {parsed_data['elevation']:.2f} degrees")

            # 두 번째 데이터와 첫 번째 데이터를 결합하여 저장
            self.collected_data.append({
                'can_id': can_id,
                'raw_data_1': first_data['raw_data_1'],
                'swapped_data_1': first_data['swapped_data_1'],
                'swapped_data_1_bin': first_data['swapped_data_1_bin'],
                'raw_data_2': raw_data,
                'swapped_data_2': swapped_data_2['swapped_data'],
                'swapped_data_2_bin': swapped_data_2['swapped_data_bin'],
                'range_m': first_data['range_m'],
                'range_m_bin': self.format_to_64bit_bin(int(first_data['range_m'] * 25)),
                'azimuth_angle': first_data['azimuth_angle'],
                'azimuth_angle_bin': self.format_to_64bit_bin(int((first_data['azimuth_angle'] / 0.16) + 511)),
                'speed_radial': first_data['speed_radial'],
                'speed_radial_bin': self.format_to_64bit_bin(int((first_data['speed_radial'] / 0.04) + 2992)),
                'rcs': parsed_data['rcs'],
                'rcs_bin': self.format_to_64bit_bin(int((parsed_data['rcs'] + 15) / 0.2)),
                'power': parsed_data['power'],
                'power_bin': self.format_to_64bit_bin(parsed_data['power']),
                'noise': parsed_data['noise'],
                'noise_bin': self.format_to_64bit_bin(int(parsed_data['noise'] / 0.5)),
                'elevation': parsed_data['elevation'],
                'elevation_bin': self.format_to_64bit_bin(int((parsed_data['elevation'] / 0.04) + 511))
            })
            # 타겟 상태 초기화
            target_state['id_401_count'] = 0
            target_state['first_data'] = None

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

    def format_to_64bit_bin(self, value):
        # 64비트 이진수로 변환하고 길이를 64비트로 고정
        return f"{value:064b}"[-64:]

    def parse_can_message(self, data):
        if len(data) != 8:
            raise ValueError("Expected 8 bytes of data")

        # Swap the data bytes and convert to 64-bit binary
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
            'reversed_64bit': bits  # 64비트 이진수로 변환된 값
        }

        return parsed_data, bit_info

    def save_data_to_file(self):
        with open('combined_data_with_bits_and_64bit.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'CAN ID', 'raw_data_1', 'swapped_data_1', 'swapped_data_1_bin', 
                'raw_data_2', 'swapped_data_2', 'swapped_data_2_bin',
                'range_m', 'range_m_bin', 'azimuth_angle', 'azimuth_angle_bin',
                'speed_radial', 'speed_radial_bin', 'rcs', 'rcs_bin', 'power', 
                'power_bin', 'noise', 'noise_bin', 'elevation', 'elevation_bin'
            ])
            for data in self.collected_data:
                writer.writerow([
                    data['can_id'],
                    data['raw_data_1'],
                    data['swapped_data_1'],
                    data['swapped_data_1_bin'],
                    data['raw_data_2'],
                    data['swapped_data_2'],
                    data['swapped_data_2_bin'],
                    data['range_m'],
                    data['range_m_bin'],
                    data['azimuth_angle'],
                    data['azimuth_angle_bin'],
                    data['speed_radial'],
                    data['speed_radial_bin'],
                    data['rcs'],
                    data['rcs_bin'],
                    data['power'],
                    data['power_bin'],
                    data['noise'],
                    data['noise_bin'],
                    data['elevation'],
                    data['elevation_bin']
                ])

        self.get_logger().info('Saved all collected CAN data, parsed data, and bit information with 64-bit data to file.')

    def destroy_node(self):
        self.bus.shutdown()
        self.get_logger().info('CAN bus has been shut down properly.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    radar_node = RadarNode()

    def signal_handler(sig, frame):
        radar_node.save_data_to_file()
        radar_node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    rclpy.spin(radar_node)

if __name__ == '__main__':
    main()

