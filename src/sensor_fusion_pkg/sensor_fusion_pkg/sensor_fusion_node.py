import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
from cv_bridge import CvBridge

class SensorFusionNode(Node):

    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.camera_sub = Subscriber(self, Image, '/camera_topic')
        self.radar_sub = Subscriber(self, PointCloud2, '/radar_topic')

        self.ts = ApproximateTimeSynchronizer(
            [self.camera_sub, self.radar_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

    def callback(self, camera_data, radar_data):
        cv_image = self.bridge.imgmsg_to_cv2(camera_data, desired_encoding='bgr8')
        self.get_logger().info(f'Received synchronized data: {len(radar_data.data)} radar points')

        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()
    rclpy.spin(sensor_fusion_node)
    sensor_fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
