import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import message_filters
import signal

class ExternalCalibrationNode(Node):
    def __init__(self):
        super().__init__('external_calibration_node')
        self.get_logger().info("Initializing ExternalCalibrationNode")
        self.bridge = CvBridge()

        # Use message filters for time synchronization
        self.camera_sub = message_filters.Subscriber(self, Image, 'camera/image_raw')
        self.radar_sub = message_filters.Subscriber(self, PointStamped, 'radar/transformed_point')

        # Message synchronization with a time tolerance of 0.1 seconds
        ts = message_filters.ApproximateTimeSynchronizer([self.camera_sub, self.radar_sub], 10, 0.1)
        ts.registerCallback(self.synchronized_callback)

        # Use deque with a maximum length to manage memory
        self.matching_points = deque(maxlen=100)
        self.camera_only_points = deque(maxlen=100)  # To store camera points when there's no match
        self.radar_only_points = deque(maxlen=100)  # To store radar points when there's no match

        # Set the threshold for reflector detection as a ROS parameter
        self.declare_parameter('reflector_threshold', 200)
        self.reflector_threshold = self.get_parameter('reflector_threshold').value

        # Add flags for calibration status and shutdown request
        self.calibration_done = False
        self.shutdown_requested = False

        # Add variables to store calibration results
        self.R_calibrated = np.eye(3)
        self.t_calibrated = np.zeros(3)
        self.rmse = None

        # Add a timer for status output
        self.create_timer(1.0, self.print_status)

        # Set up real-time 3D visualization
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.radar_scatter = self.ax.scatter([], [], [], c='r', marker='^', label='Radar Points')
        self.transformed_scatter = self.ax.scatter([], [], [], c='g', marker='o', label='Transformed Radar Points')
        self.camera_scatter = self.ax.scatter([], [], [], c='b', marker='s', label='Camera Points')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        plt.show(block=False)

    def synchronized_callback(self, camera_msg, radar_msg):
        self.get_logger().info(f"Received synchronized data: Camera timestamp: {camera_msg.header.stamp.sec}.{camera_msg.header.stamp.nanosec}, Radar timestamp: {radar_msg.header.stamp.sec}.{radar_msg.header.stamp.nanosec}")

        # Debugging: Log the radar message data
        self.get_logger().info(f"Radar Point: x={radar_msg.point.x}, y={radar_msg.point.y}, z={radar_msg.point.z}")

        cv_image = self.bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        reflector_position = self.detect_reflector(cv_image)

        radar_point = np.array([radar_msg.point.x, radar_msg.point.y, radar_msg.point.z])

        if reflector_position is not None:
            self.matching_points.append((reflector_position, radar_point))
            self.get_logger().info(f'Matched Points: Camera {reflector_position}, Radar {radar_point}')
        else:
            # If no match is found, store the points separately
            self.camera_only_points.append(reflector_position if reflector_position is not None else np.array([np.nan, np.nan, np.nan]))
            self.radar_only_points.append(radar_point)
            self.get_logger().info(f'Added Radar Point: {radar_point}')

        self.get_logger().info(f'Current Matched Points Count: {len(self.matching_points)}')
        self.get_logger().info(f'Current Matching Points: {self.matching_points}')

        if len(self.matching_points) >= 5 and not self.calibration_done:
            self.get_logger().info("Starting Calibration")
            self.calibrate()
        
        # Update the plot regardless of whether calibration is done
        self.update_plot()

    def update_plot(self):
        if not self.matching_points and not self.camera_only_points and not self.radar_only_points:
            self.get_logger().warn("No points to plot.")
            return

        if self.matching_points:
            camera_points, radar_points = zip(*self.matching_points)
            camera_points = np.array(camera_points)
            radar_points = np.array(radar_points)
            transformed_points = (self.R_calibrated @ radar_points.T).T + self.t_calibrated

            self.transformed_scatter._offsets3d = (transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2])
            self.camera_scatter._offsets3d = (camera_points[:, 0], camera_points[:, 1], camera_points[:, 2])
        else:
            camera_points = radar_points = transformed_points = np.array([])

        if self.radar_only_points:
            radar_points_only = np.array(self.radar_only_points)
            if radar_points_only.size > 0:
                self.radar_scatter._offsets3d = (radar_points_only[:, 0], radar_points_only[:, 1], radar_points_only[:, 2])

        if self.camera_only_points:
            camera_points_only = np.array(self.camera_only_points)
            if camera_points_only.size > 0:
                self.camera_scatter._offsets3d = (camera_points_only[:, 0], camera_points_only[:, 1], camera_points_only[:, 2])

        # Set plot limits to ensure data visibility
        self.ax.set_xlim([-1, 1])  # Customize this range based on your data
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])

        self.ax.relim()
        self.ax.autoscale_view()

        # Debugging: Print the points being plotted
        self.get_logger().info(f"Radar Points: {radar_points if radar_points.size > 0 else 'N/A'}")
        self.get_logger().info(f"Transformed Points: {transformed_points if transformed_points.size > 0 else 'N/A'}")
        self.get_logger().info(f"Camera Points: {camera_points if camera_points.size > 0 else 'N/A'}")

        self.fig.canvas.draw()

    def print_status(self):
        self.get_logger().info(f'Current Status: Matched Points Count = {len(self.matching_points)}, Calibration Completed = {self.calibration_done}')

    def detect_reflector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.reflector_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.get_logger().warn('Could not find contours in the image')
            return None

        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)

        if M['m00'] == 0:
            self.get_logger().warn('Invalid contour moments')
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy, 0.0])

    def calibrate(self):
        if len(self.matching_points) < 3:
            self.get_logger().warn('Not enough matched points for calibration. Collecting more data...')
            return

        def objective_function(params):
            R_matrix = R.from_rotvec(params[:3]).as_matrix()
            t_vector = params[3:]
            errors = []
            for camera_point, radar_point in self.matching_points:
                transformed_point = R_matrix @ np.array(radar_point) + t_vector
                error = np.linalg.norm(transformed_point - camera_point)
                errors.append(error)
            return errors

        initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.07])
        result = least_squares(objective_function, initial_guess, method='lm', ftol=1e-8, xtol=1e-8, max_nfev=1000)

        self.R_calibrated = R.from_rotvec(result.x[:3]).as_matrix()
        self.t_calibrated = result.x[3:]
        self.rmse = np.sqrt(np.mean(np.square(result.fun)))

        self.calibration_done = True
        self.get_logger().info(f'Calibration Completed. RMSE: {self.rmse}')
        self.get_logger().info(f'Rotation Matrix:\n{self.R_calibrated}')
        self.get_logger().info(f'Translation Vector: {self.t_calibrated}')

    def signal_handler(self):
        self.get_logger().info("Ctrl+C detected. Shutting down.")
        self.shutdown_requested = True
        plt.close(self.fig)

def main(args=None):
    rclpy.init(args=args)
    node = ExternalCalibrationNode()

    def sigint_handler(sig, frame):
        node.signal_handler()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
            plt.pause(0.01)  # Process plt events every frame
    except Exception as e:
        node.get_logger().error(f"Main loop error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

