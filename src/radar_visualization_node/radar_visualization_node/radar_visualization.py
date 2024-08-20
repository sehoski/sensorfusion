import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct

class RadarVisualizationNode(Node):
    def __init__(self):
        super().__init__('radar_visualization_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            'radar_data',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(PointCloud2, 'radar_points', 10)
        self.all_points = []  # 누적된 포인트 데이터 저장

    def listener_callback(self, msg):
        points = self.process_radar_data(msg)
        self.publish_points(points)

    def process_radar_data(self, msg):
        # 레이더 데이터 메시지를 파싱하여 포인트로 변환
        points = []
        for point in pc2.read_points(msg, skip_nans=True):
            x, y, z = point[:3]
            distance = np.sqrt(x**2 + y**2 + z**2)
            rgb = self.distance_to_rgb(distance)
            points.append([x, y, z, rgb])
        return points

    def distance_to_rgb(self, distance, max_distance=327.64):
        distance_min = 0
        distance_max = max_distance
        normalized_distance = (distance - distance_min) / (distance_max - distance_min)
        normalized_distance = np.clip(normalized_distance, 0, 1)

        red = int((1 - normalized_distance) * 255)
        blue = int(normalized_distance * 255)
        green = 0

        return struct.unpack('I', struct.pack('BBBB', blue, green, red, 255))[0]

    def publish_points(self, points):
        if not points:
            return

        self.all_points.extend(points)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'radar_frame'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        point_cloud = pc2.create_cloud(header, fields, self.all_points)
        self.publisher_.publish(point_cloud)

def main(args=None):
    rclpy.init(args=args)
    radar_node = RadarVisualizationNode()

    rclpy.spin(radar_node)

    radar_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

