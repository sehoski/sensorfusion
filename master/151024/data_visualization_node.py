import rclpy
from rclpy.node import Node
import json
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

class RadarPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('radar_point_cloud_publisher')
        self.point_cloud_publisher = self.create_publisher(PointCloud2, 'radar_point_cloud', 10)
        self.timer = self.create_timer(0.1, self.publish_point_cloud)

    def read_radar_data(self):
        data = []
        with open('/home/seho/ros2_ws/src/sensor_data/sensor_data_20241015_161611/radar_data.json', 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self.get_logger().error(f"JSON Decode Error: {e}")
        return data

    def publish_point_cloud(self):
        radar_data = self.read_radar_data()
        points = []

        for obj in radar_data:
            point_data = obj['point']
            x, y, z = point_data[0], point_data[1], point_data[2]
            intensity = point_data[4]  # rcs 값 사용

            # 포인트 데이터를 튜플로 추가 (x, y, z, intensity)
            points.append([x, y, z, intensity])

        # 포인트 데이터를 NumPy 배열로 변환
        points = np.array(points, dtype=np.float32)

        # 포인트 클라우드 메시지 생성
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()

        # PointCloud2 생성
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header = header
        point_cloud_msg.height = 1
        point_cloud_msg.width = points.shape[0]
        point_cloud_msg.is_dense = True
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 16  # 각 포인트 당 바이트 수 (x, y, z, intensity)
        point_cloud_msg.row_step = point_cloud_msg.point_step * points.shape[0]

        # PointField 설정 (x, y, z, intensity 순서)
        point_cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # 데이터를 바이너리로 변환하여 포인트 클라우드에 저장
        point_cloud_msg.data = np.asarray(points, np.float32).tobytes()

        # 포인트 클라우드 퍼블리시
        self.point_cloud_publisher.publish(point_cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    radar_point_cloud_publisher = RadarPointCloudPublisher()
    rclpy.spin(radar_point_cloud_publisher)
    radar_point_cloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

