import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='radar_package',
            executable='radar_node',
            name='radar_node'
        ),
        launch_ros.actions.Node(
            package='my_camera_pkg',
            executable='camera_node',
            name='camera_node'
        ),
        launch_ros.actions.Node(
            package='sensor_fusion_pkg',
            executable='sensor_fusion_node',
            name='sensor_fusion_node'
        ),
        launch_ros.actions.Node(
            package='sensor_fusion_pkg',
            executable='time_sync_node',
            name='time_sync_node'
        ),
    ])

if __name__ == '__main__':
    generate_launch_description()

