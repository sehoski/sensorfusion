from setuptools import setup

package_name = 'data_visualization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A package for playing back recorded sensor data',
    license='Your License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_visualization = data_visualization.data_visualization_node:main'
        ],
    },
)

