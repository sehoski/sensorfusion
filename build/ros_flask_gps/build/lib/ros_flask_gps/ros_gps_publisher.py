import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import requests

FLASK_SERVER_URL = 'http://127.0.0.1:5000/update_gps'

class GPSPublisher(Node):
    def __init__(self):
        super().__init__('gps_publisher')
        self.subscription = self.create_subscription(
            NavSatFix,
            '/gps/fix',
            self.gps_callback,
            10
        )

    def gps_callback(self, data):
        gps_data = {
            'latitude': data.latitude,
            'longitude': data.longitude,
            'altitude': data.altitude
        }
        try:
            response = requests.post(FLASK_SERVER_URL, json=gps_data)
            self.get_logger().info(f"GPS data sent to Flask server: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Failed to send GPS data: {e}")

def main(args=None):
    rclpy.init(args=args)
    gps_publisher = GPSPublisher()
    rclpy.spin(gps_publisher)
    gps_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
