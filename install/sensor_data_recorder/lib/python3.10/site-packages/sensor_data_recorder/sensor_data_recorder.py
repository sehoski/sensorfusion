import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import rosbag2_py

class SensorDataRecorder(Node):
    def __init__(self):
        super().__init__('sensor_data_recorder')
        self.get_logger().info("Sensor Data Recorder Node Started")

        self.bag = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri='sensor_data', storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        )
        self.bag.open(storage_options, converter_options)

        self.radar_subscription = self.create_subscription(
            PointCloud2, '/radar/points', self.radar_callback, 10
        )
        self.camera_subscription = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

    def radar_callback(self, msg):
        self.get_logger().info("Radar data received")
        self.bag.write('/radar/points', msg)

    def camera_callback(self, msg):
        self.get_logger().info("Camera data received")
        self.bag.write('/camera/image_raw', msg)

    def destroy_node(self):
        self.bag.close()
        self.get_logger().info("Bag file closed")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SensorDataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
