import can
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PointStamped

class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')
        self.get_logger().info("Initializing RadarNode")
        try:
            self.bus = can.interface.Bus(channel='can0', bustype='socketcan')
            self.get_logger().info("CAN bus initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CAN bus: {e}")
            raise

        self.timer = self.create_timer(0.06, self.timer_callback)
        self.collected_data = []
        self.target_states = {}
        
        # RCS 임계값 설정 (dB 단위)
        self.rcs_threshold = 20.0  # 35.0에서 20.0으로 낮춤
        # Noise 임계값 설정 (dB 단위)
        self.noise_threshold = 120.0  # 100.0에서 120.0으로 높임
        
        # 퍼블리셔 추가
        self.radar_publisher = self.create_publisher(PointStamped, 'radar/transformed_point', 10)
        self.get_logger().info("RadarNode initialization complete")

    def __del__(self):
        if hasattr(self, 'bus'):
            self.bus.shutdown()
            self.get_logger().info("CAN bus shut down")

    def timer_callback(self):
        self.get_logger().debug("Timer callback triggered")
        try:
            batch_size = 10
            for _ in range(batch_size):
                message = self.bus.recv(0.005)
                if message is not None:
                    self.get_logger().debug(f"Received CAN message: ID={message.arbitration_id}")
                    can_id = message.arbitration_id
                    raw_data = message.data.hex()
                    if 0x401 <= can_id <= 0x4FF:
                        parsed_data, bit_info = self.parse_can_message(message.data)
                        if parsed_data:
                            self.process_radar_data(parsed_data, can_id, raw_data)
                    else:
                        self.get_logger().debug(f"Ignoring message with ID: {can_id}")
                else:
                    self.get_logger().debug("No message received in this iteration")
        except AttributeError as e:
            self.get_logger().error(f"AttributeError in timer_callback: {e}")
            self.get_logger().error(f"Make sure all methods are properly defined in the class")
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")
            self.get_logger().error(f"Exception type: {type(e).__name__}")
            self.get_logger().error(f"Exception details: {str(e)}")

    def process_radar_data(self, parsed_data, can_id, raw_data):
        self.get_logger().debug(f"Processing radar data: CAN ID={can_id}")
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
                self.get_logger().info(f"Converted coordinates: x={x}, y={y}, z={z}")
                
                # PointStamped 메시지 생성 및 발행
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "radar_frame"
                point_msg.point.x = float(x)
                point_msg.point.y = float(y)
                point_msg.point.z = float(z)
                self.radar_publisher.publish(point_msg)
                self.get_logger().info(f"Published point: x={point_msg.point.x}, y={point_msg.point.y}, z={point_msg.point.z}")

                new_data = {
                    'can_id': can_id,
                    'raw_data_1': first_data['raw_data_1'],
                    'swapped_data_1': first_data['swapped_data_1'],
                    'swapped_data_1_bin': first_data['swapped_data_1_bin'],
                    'raw_data_2': raw_data,
                    'swapped_data_2': swapped_data_2['swapped_data'],
                    'swapped_data_2_bin': swapped_data_2['swapped_data_bin'],
                    'range_m': first_data['range_m'],
                    'azimuth_angle': first_data['azimuth_angle'],
                    'speed_radial': first_data['speed_radial'],
                    'rcs': parsed_data['rcs'],
                    'power': parsed_data['power'],
                    'noise': parsed_data['noise'],
                    'elevation': parsed_data['elevation'],
                    'x': x,
                    'y': y,
                    'z': z
                }

                self.collected_data.append(new_data)
                self.get_logger().info(f"New data collected: {new_data}")
            else:
                if parsed_data['rcs'] < self.rcs_threshold:
                    self.get_logger().warn(f"Data discarded due to low RCS: {parsed_data['rcs']:.2f} dB")
                if parsed_data['noise'] > self.noise_threshold:
                    self.get_logger().warn(f"Data discarded due to high noise: {parsed_data['noise']:.2f} dB")

            target_state['id_401_count'] = 0
            target_state['first_data'] = None

    def convert_to_xyz(self, range_m, azimuth_angle, elevation):
        self.get_logger().debug(f"Converting: range={range_m}, azimuth={azimuth_angle}, elevation={elevation}")
        x = range_m * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth_angle))
        y = range_m * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth_angle))
        z = range_m * np.sin(np.radians(elevation))
        self.get_logger().debug(f"Converted coordinates: x={x}, y={y}, z={z}")
        return x, y, z

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
        self.get_logger().debug("Parsing CAN message")
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

        self.get_logger().debug(f"Parsed data: {parsed_data}")
        return parsed_data, bit_info

def main(args=None):
    rclpy.init(args=args)
    
    radar_node = RadarNode()
    
    try:
        rclpy.spin(radar_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        radar_node.get_logger().error(f"Unexpected error: {e}")
    finally:
        radar_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
