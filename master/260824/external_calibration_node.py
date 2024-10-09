import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped  # Float32MultiArray 대신 PointStamped로 변경
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # GUI 환경에 맞는 백엔드 설정
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        # 3D 시각화 설정
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)
        plt.ion()
        plt.show()

        # ROS2 구독자 설정
        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10
        )

        self.radar_sub = self.create_subscription(
            PointStamped,  # Float32MultiArray 대신 PointStamped 사용
            'radar/transformed_point',
            self.radar_callback,
            10
        )

        # 시각화 데이터 저장용 변수
        self.visualization_data = {
            'radar': [],
            'camera': [],
            'transformed_radar': []
        }

    def camera_callback(self, msg):
        # 로그 메시지 추가: 데이터 확인
        self.get_logger().info(f"Camera callback triggered with timestamp: {msg.header.stamp}")
        
        # 임의의 포인트 추가 (320, 240)
        center = (320, 240)
        self.visualization_data['camera'].append(center)
        
        # 로그 메시지 추가: 시각화 데이터 확인
        self.get_logger().info(f"Camera point added: {center}")
        
        # 시각화 호출
        self.visualize_data()

    def radar_callback(self, msg):
        # 로그 메시지 추가: 데이터 확인
        self.get_logger().info(f"Radar callback triggered with data: x={msg.point.x}, y={msg.point.y}, z={msg.point.z}")
        
        # 받은 포인트 데이터를 시각화 데이터에 추가
        radar_point = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.visualization_data['radar'].append(radar_point)
        
        # 로그 메시지 추가: 시각화 데이터 확인
        self.get_logger().info(f"Radar point added: {radar_point}")
        
        # 시각화 호출
        self.visualize_data()

    def visualize_data(self):
        self.ax.clear()
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)

        if self.visualization_data['radar']:
            self.get_logger().info(f"Plotting {len(self.visualization_data['radar'])} radar points")
            self.ax.scatter(
                [point[0] for point in self.visualization_data['radar']],
                [point[1] for point in self.visualization_data['radar']],
                [point[2] for point in self.visualization_data['radar']],
                c='r', marker='o', label='Radar Points'
            )

        if self.visualization_data['camera']:
            self.get_logger().info(f"Plotting {len(self.visualization_data['camera'])} camera points")
            self.ax.scatter(
                [point[0] for point in self.visualization_data['camera']],
                [point[1] for point in self.visualization_data['camera']],
                [0] * len(self.visualization_data['camera']),  # 2D 이미지이므로 Z=0으로 설정
                c='g', marker='x', label='Camera Points'
            )

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def shutdown(self):
        print("Shutting down...")
        plt.close(self.fig)

def main(args=None):
    rclpy.init(args=args)
    calibration_node = CalibrationNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(calibration_node, timeout_sec=0.1)
            plt.pause(0.1)  # Matplotlib 이벤트 처리
    except KeyboardInterrupt:
        calibration_node.shutdown()
    finally:
        calibration_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

