import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math

LINEAR_VEL = 0.5
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
LIDAR_AVOID_DISTANCE = 0.7
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR
WALL_FOLLOW_DISTANCE = 0.3  # Distance to maintain from the wall
RIGHT_SIDE_INDEX = 270
RIGHT_FRONT_INDEX = 210
FRONT_INDEX = 180
LEFT_FRONT_INDEX = 150
LEFT_SIDE_INDEX = 90

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
        self.timer_period = 0.1
        self.cmd = Twist()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.wall_found = False  # Track if wall is found

    def listener_callback1(self, msg1):
        scan = msg1.ranges
        self.scan_cleaned = []
        
        for reading in scan:
            if reading == float('Inf'):
                self.scan_cleaned.append(3.5)  # Maximum range for LIDAR
            elif math.isnan(reading):
                self.scan_cleaned.append(0.0)  # Invalid readings as 0
            else:
                self.scan_cleaned.append(reading)

    def listener_callback2(self, msg2):
        # Odometry data if needed for more advanced localization
        pass

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

        # 2. Wall found, follow the wall on the right while avoiding front obstacles
        if front_lidar_min < SAFE_STOP_DISTANCE:
            # Obstacle ahead, turn left
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.3
            self.publisher_.publish(self.cmd)
            self.get_logger().info('Obstacle ahead, turning left...')
        elif right_lidar_min < WALL_FOLLOW_DISTANCE - 0.1:
            # Too close to the wall, turn left slightly
            self.cmd.linear.x = LINEAR_VEL * 0.5
            self.cmd.angular.z = 0.3
            self.publisher_.publish(self.cmd)
            self.get_logger().info('Too close to wall, adjusting left...')
        elif right_lidar_min > WALL_FOLLOW_DISTANCE + 0.1:
            # Too far from the wall, turn right slightly
            self.cmd.linear.x = LINEAR_VEL * 0.5
            self.cmd.angular.z = -0.3
            self.publisher_.publish(self.cmd)
            self.get_logger().info('Too far from wall, adjusting right...')
        else:
            # Maintain distance from the wall and move forward
            self.cmd.linear.x = LINEAR_VEL
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            self.get_logger().info('Following the wall...')

def main(args=None):
    rclpy.init(args=args)
    wall_follower_node = WallFollower()
    rclpy.spin(wall_follower_node)
    wall_follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
