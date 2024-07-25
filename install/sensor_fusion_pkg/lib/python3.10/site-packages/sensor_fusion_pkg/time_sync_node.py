import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2

class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('time_sync_node')

        self.camera_sub = Subscriber(self, Image, '/camera_topic')
        self.radar_sub = Subscriber(self, PointCloud2, '/radar_topic')

        self.ts = ApproximateTimeSynchronizer(
            [self.camera_sub, self.radar_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

    def callback(self, camera_data, radar_data):
        self.get_logger().info('Synchronized messages received')
        cv_image = self.bridge.imgmsg_to_cv2(camera_data, desired_encoding='bgr8')
        self.get_logger().info(f'Received radar data with {len(radar_data.data)} points')
        cv2.imshow('Camera', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    time_sync_node = TimeSyncNode()
    rclpy.spin(time_sync_node)
    time_sync_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

