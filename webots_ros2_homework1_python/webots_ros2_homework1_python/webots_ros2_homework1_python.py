import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import keyboard  # New import to handle keypresses

# Constants
LINEAR_VEL = 0.22
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
LIDAR_AVOID_DISTANCE = 0.35  # Follow wall closer
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR
TURN_SPEED = 0.5  # Faster turning when adjusting
RIGHT_SIDE_INDEX = 270
RIGHT_FRONT_INDEX = 210
LEFT_FRONT_INDEX = 150
LEFT_SIDE_INDEX = 90
DISTANCE_THRESHOLD = 0.001  # Threshold for determining the most distant points
APARTMENT_IMAGE_PATH = "apartment.png"

class RandomWalk(Node):

    def __init__(self):
        super().__init__('random_walk_node')
        self.scan_cleaned = []
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscriber1 = self.create_subscription(
            LaserScan, '/scan', self.listener_callback1,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber2 = self.create_subscription(
            Odometry, '/odom', self.listener_callback2,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.odom_data = None
        self.pose_history = []
        self.total_distance = 0.0
        self.distant_points = {"top_left": (-float('inf'), float('inf')),
                               "top_right": (float('inf'), float('inf')),
                               "bottom_left": (-float('inf'), -float('inf')),
                               "bottom_right": (float('inf'), -float('inf'))}
        self.cmd = Twist()
        self.timer = self.create_timer(0.5, self.timer_callback)

    def listener_callback1(self, msg1):
        scan = msg1.ranges
        self.scan_cleaned = [3.5 if reading == float('Inf') else (0.0 if math.isnan(reading) else reading) for reading in scan]

    def listener_callback2(self, msg2):
        position = msg2.pose.pose.position
        if self.odom_data is not None:
            self.total_distance += math.sqrt((position.x - self.odom_data.x)**2 + (position.y - self.odom_data.y)**2)
        self.odom_data = position
        self.pose_history.append((position.x, position.y))
        self.update_distant_points(position)

    def update_distant_points(self, position):
        # Update the most distant points in each zone
        if position.x < 0 and position.y > 0:  # Top left zone
            if position.x < self.distant_points["top_left"][0]:
                self.distant_points["top_left"] = (position.x, position.y)
        elif position.x > 0 and position.y > 0:  # Top right zone
            if position.x > self.distant_points["top_right"][0]:
                self.distant_points["top_right"] = (position.x, position.y)
        elif position.x < 0 and position.y < 0:  # Bottom left zone
            if position.x < self.distant_points["bottom_left"][0]:
                self.distant_points["bottom_left"] = (position.x, position.y)
        elif position.x > 0 and position.y < 0:  # Bottom right zone
            if position.x > self.distant_points["bottom_right"][0]:
                self.distant_points["bottom_right"] = (position.x, position.y)

    def timer_callback(self):
        if not self.scan_cleaned:
            return

        left_lidar_min = min(self.scan_cleaned[LEFT_SIDE_INDEX:LEFT_FRONT_INDEX])
        right_lidar_min = min(self.scan_cleaned[RIGHT_FRONT_INDEX:RIGHT_SIDE_INDEX])
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])

        if front_lidar_min < SAFE_STOP_DISTANCE:
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
        elif front_lidar_min < LIDAR_AVOID_DISTANCE:
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = TURN_SPEED if right_lidar_min > left_lidar_min else -TURN_SPEED
        else:
            self.cmd.linear.x = LINEAR_VEL
            self.cmd.angular.z = 0.0

        self.publisher_.publish(self.cmd)

    def save_results(self):
        # Save path history and distant points
        with open('trial_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
            for point in self.pose_history:
                writer.writerow([point[0], point[1]])

        # Log total distance and distant points
        self.get_logger().info(f'Total distance: {self.total_distance}')
        for zone, coords in self.distant_points.items():
            self.get_logger().info(f'{zone.capitalize()} - Most distant point: {coords}')

    def plot_path(self):
        apartment_img = Image.open(APARTMENT_IMAGE_PATH)
        plt.imshow(apartment_img, extent=[-10, 10, -10, 10])  # Adjust extent according to the map size

        data = np.array(self.pose_history)
        plt.plot(data[:, 0], data[:, 1], label='Trial Path')

        plt.legend()
        plt.title('Robot Path for Trial')
        plt.savefig('robot_path.png')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    random_walk_node = RandomWalk()

    try:
        # Run the trial until 'E' key is pressed
        print("Press 'E' to end the trial.")
        while not keyboard.is_pressed('e'):
            rclpy.spin_once(random_walk_node)
    except KeyboardInterrupt:
        pass  # Handle ctrl+c to exit

    # Save the results and plot the path
    random_walk_node.save_results()
    random_walk_node.plot_path()

    random_walk_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
