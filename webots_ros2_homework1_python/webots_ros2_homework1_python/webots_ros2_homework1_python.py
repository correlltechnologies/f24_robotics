import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math

LINEAR_VEL = 0.22
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
LIDAR_AVOID_DISTANCE = 0.7
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR

class RandomWalk(Node):

    def __init__(self):
        super().__init__('random_walk_node')
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
        
        self.scan_cleaned = []
        self.target_zones = [
            {'translation': (7.6, 3.5), 'rotation': (0, 0, 1, 0)},
            {'translation': (5.2, 6.0), 'rotation': (0, 0, 1, 0)},
            {'translation': (1.8, 6.3), 'rotation': (0, 0, 1, -1.57)},
            {'translation': (0.96, 0.492), 'rotation': (0, 0, 1, 0)}
        ]
        self.visited_zones = []
        self.current_target = None
        self.cmd = Twist()
        self.timer = self.create_timer(0.5, self.timer_callback)

    def listener_callback1(self, msg1):
        self.scan_cleaned = [reading if reading != float('Inf') and not math.isnan(reading) else 3.5 for reading in msg1.ranges]

    def listener_callback2(self, msg2):
        position = msg2.pose.pose.position
        self.current_position = (position.x, position.y)
        self.get_logger().info('Current position: {}'.format(self.current_position))

        # Determine the next target zone if not set
        if self.current_target is None:
            self.current_target = self.get_closest_zone()

    def get_closest_zone(self):
        closest_zone = None
        min_distance = float('inf')

        for zone in self.target_zones:
            zone_position = zone['translation']
            distance = math.sqrt((self.current_position[0] - zone_position[0]) ** 2 +
                                 (self.current_position[1] - zone_position[1]) ** 2)
            if distance < min_distance and zone not in self.visited_zones:
                min_distance = distance
                closest_zone = zone

        return closest_zone

    def timer_callback(self):
        if not self.scan_cleaned:
            return
        
        front_lidar_min = min(self.scan_cleaned[90:270])  # Adjust indices as needed
        
        if front_lidar_min < SAFE_STOP_DISTANCE:
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            return

        if self.current_target:
            target_x, target_y = self.current_target['translation']
            distance_to_target = math.sqrt((self.current_position[0] - target_x) ** 2 +
                                            (self.current_position[1] - target_y) ** 2)
            
            # Mark zone as visited if close enough
            if distance_to_target < 0.1:  # You can adjust the threshold
                self.visited_zones.append(self.current_target)
                self.get_logger().info('Visited zone: {}'.format(self.current_target['translation']))
                self.current_target = self.get_closest_zone()  # Get the next target zone
                if not self.current_target:
                    self.get_logger().info('All zones visited!')
                    return  # Stop further processing

            angle_to_target = math.atan2(target_y - self.current_position[1], target_x - self.current_position[0])
            current_orientation = math.atan2(self.current_position[1], self.current_position[0])  # Approximation
            
            # Turn towards the target zone
            if abs(angle_to_target - current_orientation) > 0.1:
                self.cmd.angular.z = 0.3 if angle_to_target > current_orientation else -0.3
                self.cmd.linear.x = 0.0
            else:
                self.cmd.linear.x = LINEAR_VEL
                self.cmd.angular.z = 0.0
            
            self.publisher_.publish(self.cmd)

        self.get_logger().info('Moving towards: {}'.format(self.current_target['translation']))


def main(args=None):
    rclpy.init(args=args)
    random_walk_node = RandomWalk()
    rclpy.spin(random_walk_node)
    random_walk_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
