import can
import rclpy
from rclpy.node import Node
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
from collections import deque

class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')
        self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
        self.timer = self.create_timer(0.06, self.timer_callback)
        self.max_data_points = 100  # 최대 데이터 포인트 수 설정
        self.collected_data = deque(maxlen=self.max_data_points)
        self.target_states = {}
        
        # RCS 임계값 설정 (dB 단위)
        self.rcs_threshold = 35.0
        # Noise 임계값 설정 (dB 단위)
        self.noise_threshold = 85.0

        self.fig = plt.figure(figsize=(12, 5))
        
        # Polar plot
        self.ax_polar = self.fig.add_subplot(121, projection='polar')
        self.scatter_polar = self.ax_polar.scatter([], [], c=[], cmap='cool')
        self.ax_polar.set_ylim(0, 20)
        self.ax_polar.set_theta_zero_location('N')
        self.ax_polar.set_theta_direction(-1)
        self.colorbar_polar = self.fig.colorbar(self.scatter_polar, ax=self.ax_polar)
        self.colorbar_polar.set_label('Speed (m/s)')
        self.ax_polar.set_title('Polar Visualization')

        # 3D position plot
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.scatter_3d = self.ax_3d.scatter([], [], [], c='blue', marker='o')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Position Visualization')

        plt.tight_layout()
        plt.ion()
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
            swapped_data_1 = self.swap_and_convert_to_64bit(raw_data)
            target_state['first_data'] = {
                'range_m': parsed_data['range_m'],
                'azimuth_angle': parsed_data['azimuth_angle'],
                'speed_radial': parsed_data['speed_radial'],
                'raw_data_1': raw_data,
                'swapped_data_1': swapped_data_1['swapped_data'],
                'swapped_data_1_bin': swapped_data_1['swapped_data_bin']
            }
        elif target_state['id_401_count'] == 2:
            swapped_data_2 = self.swap_and_convert_to_64bit(raw_data)
            first_data = target_state['first_data']

            if parsed_data['rcs'] >= self.rcs_threshold and parsed_data['noise'] <= self.noise_threshold:
                # 좌표 변환
                x, y, z = self.convert_to_xyz(first_data['range_m'], first_data['azimuth_angle'], parsed_data['elevation'])

                new_data = {
                    'can_id': can_id,
                    'range_m': first_data['range_m'],
                    'azimuth_angle': first_data['azimuth_angle'],
                    'speed_radial': first_data['speed_radial'],
                    'elevation': parsed_data['elevation'],
                    'x': x,
                    'y': y,
                    'z': z
                }

                self.collected_data.append(new_data)
                self.update_visualization()
            else:
                if parsed_data['rcs'] < self.rcs_threshold:
                    print(f"Data discarded due to low RCS: {parsed_data['rcs']:.2f} dB")
                if parsed_data['noise'] > self.noise_threshold:
                    print(f"Data discarded due to high noise: {parsed_data['noise']:.2f} dB")

            target_state['id_401_count'] = 0
            target_state['first_data'] = None

    def convert_to_xyz(self, range_m, azimuth_angle, elevation):
        x = range_m * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth_angle))
        y = range_m * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth_angle))
        z = range_m * np.sin(np.radians(elevation))
        return x, y, z

    def update_visualization(self):
        if not self.collected_data:
            return

        # Polar plot update
        theta = [np.radians(data['azimuth_angle']) for data in self.collected_data]
        r = [data['range_m'] for data in self.collected_data]
        speed = [data['speed_radial'] for data in self.collected_data]

        self.scatter_polar.set_offsets(np.column_stack((theta, r)))
        self.scatter_polar.set_array(np.array(speed))
        self.scatter_polar.set_clim(vmin=min(speed), vmax=max(speed))

        # 3D position plot update
        x_data = [data['x'] for data in self.collected_data]
        y_data = [data['y'] for data in self.collected_data]
        z_data = [data['z'] for data in self.collected_data]
        
        self.scatter_3d._offsets3d = (x_data, y_data, z_data)
        
        # Adjust 3D plot limits
        max_range = np.max([np.max(np.abs(x_data)),
                            np.max(np.abs(y_data)),
                            np.max(np.abs(z_data))])
        self.ax_3d.set_xlim(-max_range, max_range)
        self.ax_3d.set_ylim(-max_range, max_range)
        self.ax_3d.set_zlim(-max_range, max_range)

        plt.draw()
        plt.pause(0.01)

    def swap_and_convert_to_64bit(self, raw_data):
        byte_array = bytearray.fromhex(raw_data)
        swapped_byte_array = byte_array[::-1]
        swapped_data_bin = ''.join(f"{byte:08b}" for byte in swapped_byte_array)[:64]
        swapped_data_hex = swapped_byte_array.hex()
        return {
            'swapped_data': swapped_data_hex,
            'swapped_data_bin': swapped_data_bin
        }

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

def main(args=None):
    rclpy.init(args=args)
    
    radar_node = RadarNode()
    
    try:
        rclpy.spin(radar_node)
    except KeyboardInterrupt:
        pass
    finally:
        radar_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
