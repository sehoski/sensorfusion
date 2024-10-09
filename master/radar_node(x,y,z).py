import can
import rclpy
from rclpy.node import Node
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.timer = self.create_timer(0.06, self.timer_callback)
        self.collected_data = []
        self.target_states = {}

        # 실시간 시각화를 위한 초기화
        self.fig = plt.figure(figsize=(12, 6))

        # 3D 플롯 생성
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.sc = self.ax_3d.scatter([], [], [], c='blue', marker='o')

        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('Real-time 3D Position Visualization')

        # range, azimuth, elevation 플롯 생성
        self.ax_2d = self.fig.add_subplot(122, projection='3d')
        self.sc_2d = self.ax_2d.scatter([], [], [], c='red', marker='o')

        self.ax_2d.set_xlabel('Range (m)')
        self.ax_2d.set_ylabel('Azimuth (degrees)')
        self.ax_2d.set_zlabel('Elevation (degrees)')
        self.ax_2d.set_title('Range, Azimuth, Elevation Visualization')

        plt.ion()  # 인터랙티브 모드 켜기
        plt.show()

    def timer_callback(self):
        try:
            batch_size = 10
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    can_id = message.arbitration_id
                    raw_data = message.data.hex()
                    if 0x401 <= can_id <= 0x4FF:
                        parsed_data, bit_info = self.parse_can_message(message.data)
                        if parsed_data:
                            self.process_radar_data(parsed_data, can_id, raw_data)
                            self.update_plot()
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def process_radar_data(self, parsed_data, can_id, raw_data):
        if can_id not in self.target_states:
            self.target_states[can_id] = {
                'id_401_count': 0,
                'first_data': None
            }

        target_state = self.target_states[can_id]
        target_state['id_401_count'] += 1

        if target_state['id_401_count'] == 1:
            # 첫 번째 데이터: range_m, azimuth_angle, speed_radial
            target_state['first_data'] = {
                'range_m': parsed_data['range_m'],
                'azimuth_angle': parsed_data['azimuth_angle'],
                'speed_radial': parsed_data['speed_radial'],
            }
        elif target_state['id_401_count'] == 2:
            # 두 번째 데이터: rcs, power, noise, elevation
            first_data = target_state['first_data']

            # 좌표 변환
            x, y, z = self.convert_to_xyz(first_data['range_m'], first_data['azimuth_angle'], parsed_data['elevation'])

            # 데이터를 추가하여 시각화를 위해 저장
            self.collected_data.append({
                'can_id': can_id,
                'x': x,
                'y': y,
                'z': z,
                'range_m': first_data['range_m'],
                'azimuth_angle': first_data['azimuth_angle'],
                'elevation': parsed_data['elevation'],
                'speed_radial': first_data['speed_radial'],
                'rcs': parsed_data['rcs'],
                'power': parsed_data['power'],
                'noise': parsed_data['noise']
            })

            target_state['id_401_count'] = 0
            target_state['first_data'] = None

    def convert_to_xyz(self, range_m, azimuth_angle, elevation):
        # 거리와 각도를 사용하여 x, y, z 좌표 계산
        x = range_m * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth_angle))
        y = range_m * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth_angle))
        z = range_m * np.sin(np.radians(elevation))
        return x, y, z

    def update_plot(self):
        # 시각화를 업데이트
        if not self.collected_data:
            return

        x_data = [data['x'] for data in self.collected_data]
        y_data = [data['y'] for data in self.collected_data]
        z_data = [data['z'] for data in self.collected_data]

        range_data = [data['range_m'] for data in self.collected_data]
        azimuth_data = [data['azimuth_angle'] for data in self.collected_data]
        elevation_data = [data['elevation'] for data in self.collected_data]

        # 3D 위치 플롯 업데이트
        self.sc._offsets3d = (x_data, y_data, z_data)
        
        # range, azimuth, elevation 플롯 업데이트
        self.sc_2d._offsets3d = (range_data, azimuth_data, elevation_data)

        plt.draw()
        plt.pause(0.01)  # 짧은 시간 대기하여 플롯 업데이트 반영

    def swap_and_convert_to_64bit(self, raw_data):
        byte_array = bytearray.fromhex(raw_data)
        swapped_byte_array = byte_array[::-1]
        swapped_data_bin = ''.join(f"{byte:08b}" for byte in swapped_byte_array)[:64]
        swapped_data_hex = swapped_byte_array.hex()
        return {
            'swapped_data': swapped_data_hex,
            'swapped_data_bin': swapped_data_bin
        }

    def format_to_64bit_bin(self, value):
        return f"{value:064b}"[-64:]

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
        self.bus.shutdown()
        self.get_logger().info('CAN bus has been shut down properly.')
        super().destroy_node()

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

