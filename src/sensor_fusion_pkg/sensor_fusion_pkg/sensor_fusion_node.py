import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2

class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('time_sync_node')

        self.get_logger().info('Initializing TimeSyncNode')

        self.camera_sub = Subscriber(self, Image, 'camera/image_raw')
        self.radar_sub = Subscriber(self, PointCloud2, 'radar/points')

        # 슬로프 값을 0.1로 조정하여 더 엄격한 동기화 시도
        self.ts = ApproximateTimeSynchronizer(
            [self.camera_sub, self.radar_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        
        self.get_logger().info('Subscribers initialized and ApproximateTimeSynchronizer set')

    def callback(self, camera_data, radar_data):
        self.get_logger().info('Callback function called')  # 콜백 호출 확인

        camera_time = camera_data.header.stamp
        radar_time = radar_data.header.stamp

        # Log the timestamps
        self.get_logger().info(f'Camera time: {camera_time.sec}.{camera_time.nanosec}')
        self.get_logger().info(f'Radar time: {radar_time.sec}.{radar_time.nanosec}')
        
        # Calculate time difference
        time_diff_sec = abs(camera_time.sec - radar_time.sec)
        time_diff_nsec = abs(camera_time.nanosec - radar_time.nanosec) / 1e9
        total_time_diff = time_diff_sec + time_diff_nsec
        
        # Log the time difference
        self.get_logger().info(f'Time difference: {total_time_diff} seconds')

        # Process synchronized messages
        self.get_logger().info('Synchronized messages received')
        cv_image = self.bridge.imgmsg_to_cv2(camera_data, desired_encoding='bgr8')
        self.get_logger().info(f'Received radar data with {len(radar_data.data)} points')
        cv2.imshow('Camera', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    time_sync_node = TimeSyncNode()
    try:
        rclpy.spin(time_sync_node)
    except KeyboardInterrupt:
        pass
    finally:
        time_sync_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

