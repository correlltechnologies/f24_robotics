import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math
import csv

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

class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower_node')
        self.scan_cleaned = []
        self.turtlebot_moving = False
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
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
        self.wall_found = False  # Track if wall is found
        # Add an iteration counter and a constant N
        self.iteration_counter = 0
        self.N = 50  # Change this to your desired number of iterations

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

    def timer_callback(self):
        if not self.scan_cleaned:
            return

        right_lidar_min = min(self.scan_cleaned[RIGHT_FRONT_INDEX:RIGHT_SIDE_INDEX])
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])
    
        # No wall on the right, move forward until wall is detected
        if not self.wall_found:
            if right_lidar_min > WALL_FOLLOW_DISTANCE:
                self.cmd.linear.x = LINEAR_VEL
                self.cmd.angular.z = 0.0
                self.publisher_.publish(self.cmd)
                self.get_logger().info('Searching for wall...')
            else:
                self.wall_found = True  # Wall found
                self.get_logger().info('Wall detected! Starting to follow.')
            return

        # Check if robot is stuck
        if front_lidar_min < SAFE_STOP_DISTANCE:
            self.stuck_counter += 1

            # If stuck for number of iterations, back up
            if self.stuck_counter > 10:
                self.cmd.linear.x = -0.5  # Back up
                self.cmd.angular.z = 0.0
                self.publisher_.publish(self.cmd)
                self.get_logger().info('Stuck! Backing up...')
                self.stuck_counter = 0  # Reset stuck counter after backing up
            else:
                # Turn left when facing obstacle
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.3
                self.publisher_.publish(self.cmd)
                self.get_logger().info('Obstacle ahead, turning left...')
            return
        else:
            # Reset stuck counter if no obstacle in front
            self.stuck_counter = 0

        # Wall found, follow the wall on the right
        if right_lidar_min < WALL_FOLLOW_DISTANCE - 0.1:
            # Too close to the wall, turn left
            self.cmd.linear.x = LINEAR_VEL * 0.5
            self.cmd.angular.z = 0.3
            self.publisher_.publish(self.cmd)
            if self.iteration_counter >= self.N:
                self.save_results()
                self.iteration_counter = 0
                self.get_logger().info('Too close to wall, adjusting left...')
        elif right_lidar_min > WALL_FOLLOW_DISTANCE + 0.1:
            # Too far from the wall, turn right
            self.cmd.linear.x = LINEAR_VEL * 0.5
            self.cmd.angular.z = -0.3
            self.publisher_.publish(self.cmd)
            if self.iteration_counter >= self.N:
                self.save_results()
                self.iteration_counter = 0
                self.get_logger().info('Too far from wall, adjusting right...')
        else:
            # Continue moving forward
            self.cmd.linear.x = LINEAR_VEL
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            if self.iteration_counter >= self.N:
                self.save_results()
                self.iteration_counter = 0
                self.get_logger().info('Following the wall...')
                
        self.iteration_counter += 1
        

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
