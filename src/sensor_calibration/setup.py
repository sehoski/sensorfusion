from setuptools import setup

package_name = 'sensor_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Example package for sensor calibration',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'external_calibration_node = sensor_calibration.external_calibration_node:main'
        ],
    },
)

