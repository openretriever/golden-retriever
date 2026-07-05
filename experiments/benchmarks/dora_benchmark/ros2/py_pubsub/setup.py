from setuptools import setup

package_name = 'py_pubsub'

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


    maintainer='OpenRetriever Maintainers',
    maintainer_email='maintainers@openretriever.org',
    description='ROS 2 Python pub-sub benchmark used by Golden Retriever experiments',
    license='Apache-2.0',
    tests_require=['pytest'],


    entry_points={
        'console_scripts': [
                'talker = py_pubsub.publisher_member_function:main',
                'listener = py_pubsub.subscriber_member_function:main',
        ],
},

)
