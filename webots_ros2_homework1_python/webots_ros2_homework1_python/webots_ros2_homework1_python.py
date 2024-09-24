import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf_transformations import euler_from_quaternion
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image

LINEAR_VEL = 0.2
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
LIDAR_AVOID_DISTANCE = 0.7
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR
WALL_FOLLOW_DISTANCE = 0.5  # Distance to maintain from the wall
RIGHT_SIDE_INDEX = 270
RIGHT_FRONT_INDEX = 210
FRONT_INDEX = 180
LEFT_FRONT_INDEX = 150
LEFT_SIDE_INDEX = 90
DISTANCE_THRESHOLD = 0.001  # Threshold for determining the most distant points
APARTMENT_IMAGE_PATH = "/home/quinn/f24_robotics/Homework1/apartment.png"

class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower_node')
        self.scan_cleaned = []
        self.turtlebot_moving = False
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscriptions for LaserScan and Odometry
        self.subscriber1 = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback1,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.subscriber2 = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback2,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.odom_data = None
        self.pose_history = []
        self.total_distance = 0.0
        self.distant_points = {"top_left": (-float('inf'), float('inf')), "top_right": (float('inf'), float('inf')), "bottom_left": (-float('inf'), -float('inf')), "bottom_right": (float('inf'), -float('inf'))}
        self.timer_period = 0.1
        self.cmd = Twist()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.wall_found = False
        self.iteration_counter = 0
        self.N = 50

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.global_frame = 'map'  # Global frame, usually 'map' if AMCL is running
        self.odom_frame = 'odom'   # Local frame

    def listener_callback1(self, msg1):
        scan = msg1.ranges
        self.scan_cleaned = [3.5 if reading == float('Inf') else (0.0 if math.isnan(reading) else reading) for reading in scan]

    def listener_callback2(self, msg2):
        try:
            # Try to lookup the transform from odom to the global map frame
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(self.global_frame, self.odom_frame, now)

            # Extract the translation and rotation from the transform
            position = transform.transform.translation
            rotation = transform.transform.rotation
            orientation = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
            
            # Log the global position (x, y) and the yaw angle
            global_x = position.x
            global_y = position.y
            yaw = orientation[2]  # Extract yaw from quaternion
            
            # Append to pose history
            self.pose_history.append((global_x, global_y))

            # Update the total distance traveled
            if self.odom_data is not None:
                self.total_distance += math.sqrt((global_x - self.odom_data[0])**2 + (global_y - self.odom_data[1])**2)
            self.odom_data = (global_x, global_y)

            # Update the most distant points in the zones
            self.update_distant_points(global_x, global_y)

        except Exception as e:
            self.get_logger().warn(f"Could not transform between {self.global_frame} and {self.odom_frame}: {str(e)}")


    def timer_callback(self):
        if not self.scan_cleaned:
            return  # No data yet

        right_lidar_min = min(self.scan_cleaned[RIGHT_FRONT_INDEX:RIGHT_SIDE_INDEX])
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])
    
        # 1. No wall on the right -> Move forward until wall is detected
        if not self.wall_found:
            if right_lidar_min > WALL_FOLLOW_DISTANCE:
                self.cmd.linear.x = LINEAR_VEL
                self.cmd.angular.z = 0.0
                self.publisher_.publish(self.cmd)
                self.get_logger().info('Searching for wall...')
            else:
                self.wall_found = True  # Wall found, now start wall-following
                self.get_logger().info('Wall detected! Starting to follow.')
            return

        # Check if robot is stuck (e.g., close obstacle in front for too long)
        if front_lidar_min < SAFE_STOP_DISTANCE:
            # Increment stuck counter
            self.stuck_counter += 1

            # If stuck for multiple consecutive checks, back up
            if self.stuck_counter > 10:  # Customize how many cycles determine 'stuck'
                self.cmd.linear.x = -0.1  # Back up slowly
                self.cmd.angular.z = 0.0
                self.publisher_.publish(self.cmd)
                self.get_logger().info('Stuck! Backing up...')
                self.stuck_counter = 0  # Reset stuck counter after backing up
            else:
                # Just turn left when facing obstacle
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.3
                self.publisher_.publish(self.cmd)
                self.get_logger().info('Obstacle ahead, turning left...')
            return
        else:
            # Reset stuck counter if no obstacle in front
            self.stuck_counter = 0

        # 2. Wall found, follow the wall on the right while avoiding front obstacles
        if right_lidar_min < WALL_FOLLOW_DISTANCE - 0.1:
            # Too close to the wall, turn left slightly
            self.cmd.linear.x = LINEAR_VEL * 0.5
            self.cmd.angular.z = 0.3
            self.publisher_.publish(self.cmd)
            if self.iteration_counter >= self.N:
                # Save the results and plot the path every N iterations
                self.save_results()
                self.plot_path()
            
                # Reset the counter
                self.iteration_counter = 0
                self.get_logger().info('Too close to wall, adjusting left...')
        elif right_lidar_min > WALL_FOLLOW_DISTANCE + 0.1:
            # Too far from the wall, turn right slightly
            self.cmd.linear.x = LINEAR_VEL * 0.5
            self.cmd.angular.z = -0.3
            self.publisher_.publish(self.cmd)
            if self.iteration_counter >= self.N:
                # Save the results and plot the path every N iterations
                self.save_results()
                self.plot_path()
            
                # Reset the counter
                self.iteration_counter = 0
                self.get_logger().info('Too far from wall, adjusting right...')
        else:
            # Maintain distance from the wall and move forward
            self.cmd.linear.x = LINEAR_VEL
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            # Check if the counter has reached N
            if self.iteration_counter >= self.N:
                # Save the results and plot the path every N iterations
                self.save_results()
                self.plot_path()
            
                # Reset the counter
                self.iteration_counter = 0
                self.get_logger().info('Following the wall...')
        
        # Increment the iteration counter
        self.iteration_counter += 1
    
    def update_distant_points(self, x, y):
        # Update the most distant points in each zone
        if x < 0 and y > 0:  # Top left zone
            if x < self.distant_points["top_left"][0]:
                self.distant_points["top_left"] = (x, y)
        elif x > 0 and y > 0:  # Top right zone
            if x > self.distant_points["top_right"][0]:
                self.distant_points["top_right"] = (x, y)
        elif x < 0 and y < 0:  # Bottom left zone
            if x < self.distant_points["bottom_left"][0]:
                self.distant_points["bottom_left"] = (x, y)
        elif x > 0 and y < 0:  # Bottom right zone
            if x > self.distant_points["bottom_right"][0]:
                self.distant_points["bottom_right"] = (x, y)

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

        # Assuming the image has a size of 20x20 units (you may need to adjust this based on actual map size)
        extent = [-10, 10, -10, 10]  # Example extent for a 20x20 apartment

        # Load and show the apartment image
        plt.imshow(apartment_img, extent=extent)

        # Convert the pose history to a numpy array for easier manipulation
        data = np.array(self.pose_history)

        # Scale the x and y axes by their respective scaling factors
        scale_x = 3.134  # Scaling factor for x-axis
        scale_y = 1.393  # Scaling factor for y-axis
        scaled_x = data[:, 1] * scale_x  # Scale the y-values (which become x in the plot)
        scaled_y = -data[:, 0] * scale_y  # Scale the x-values (which become y in the plot) and invert

        # Plot the adjusted path
        plt.plot(scaled_x, scaled_y, label='Trial Path', color='red', linewidth=2)

        # Add legend and title
        plt.legend()
        plt.title('Robot Path for Trial')

        # Save the plot as an image
        plt.savefig('robot_path.png')

        # Clear the figure to free memory
        plt.clf()


def main(args=None):
    rclpy.init(args=args)
    wall_follower_node = WallFollower()

    try:
        rclpy.spin(wall_follower_node)
    except KeyboardInterrupt:
        pass

    wall_follower_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
