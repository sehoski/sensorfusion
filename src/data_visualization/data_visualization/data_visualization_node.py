import rclpy
from rclpy.node import Node
import json
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import time

class RadarPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('radar_point_cloud_publisher')
        self.point_cloud_publisher = self.create_publisher(PointCloud2, 'radar_point_cloud', 10)
        self.timer = self.create_timer(0.1, self.publish_point_cloud)
        
        # 객체의 상태를 저장하는 딕셔너리 (id: [x, y, z, intensity, timestamp])
        self.active_objects = {}
        self.file_path = '/home/seho/ros2_ws/src/sensor_data/sensor_data_20241015_161611/radar_data.json'
        self.last_read_position = 0
        self.expiration_time = 2.0  # 초 단위로 객체가 사라질 시간

    def get_new_radar_data(self):
        """새로운 데이터를 파일에서 가져오는 함수."""
        new_data = []
        with open(self.file_path, 'r') as f:
            f.seek(self.last_read_position)
            for line in f:
                try:
                    new_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self.get_logger().error(f"JSON Decode Error: {e}")
            self.last_read_position = f.tell()  # 현재 위치 저장
        return new_data

    def publish_point_cloud(self):
        radar_data = self.get_new_radar_data()
        current_time = time.time()
        
        # 새로운 데이터 업데이트
        for obj in radar_data:
            obj_id = obj['can_id']
            point_data = obj['point']
            x, y, z = point_data[0], point_data[1], point_data[2]
            intensity = point_data[4]  # rcs 값 사용
            self.active_objects[obj_id] = [x, y, z, intensity, current_time]  # 객체 정보 업데이트

        # 활성 객체 중 유효 시간 초과한 객체 삭제
        self.active_objects = {
            obj_id: data
            for obj_id, data in self.active_objects.items()
            if current_time - data[4] < self.expiration_time
        }

        # 유효한 객체들로 포인트 생성
        points = []
        for data in self.active_objects.values():
            x, y, z, intensity = data[:4]
            points.append([x, y, z, intensity])

        points = np.array(points, dtype=np.float32)

        # 포인트 클라우드 메시지 생성
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()

        point_cloud_msg = PointCloud2()
        point_cloud_msg.header = header
        point_cloud_msg.height = 1
        point_cloud_msg.width = points.shape[0]
        point_cloud_msg.is_dense = True
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 16
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
        if points.size > 0:
            self.point_cloud_publisher.publish(point_cloud_msg)
            self.get_logger().info(f"Published {points.shape[0]} points")
        else:
            self.get_logger().info("No active objects detected")

def main(args=None):
    rclpy.init(args=args)
    radar_point_cloud_publisher = RadarPointCloudPublisher()
    rclpy.spin(radar_point_cloud_publisher)
    radar_point_cloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

