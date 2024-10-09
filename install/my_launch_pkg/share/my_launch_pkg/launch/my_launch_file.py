import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='radar_package',
            executable='radar_node',
            name='radar_node',
            output='screen'  # 로그 출력 설정
        ),
        launch_ros.actions.Node(
            package='my_camera_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen'  # 로그 출력 설정
        ),
        launch_ros.actions.Node(
            package='sensor_fusion_pkg',
            executable='time_sync_node',
            name='time_sync_node',
            output='screen'  # 로그 출력 설정
        ),
    ])

