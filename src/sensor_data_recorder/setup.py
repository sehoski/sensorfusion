from setuptools import setup

package_name = 'sensor_data_recorder'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Description of the package',
    license='License type',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_data_recorder_node = sensor_data_recorder.sensor_data_recorder:main',
        ],
    },
)

