from setuptools import find_packages, setup

package_name = 'benchmark_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='ROS Python benchmarking for Retriever',
    entry_points={
        'console_scripts': [
            'publisher = benchmark_python.publisher:main',
            'subscriber = benchmark_python.subscriber:main',
        ],
    },
)
