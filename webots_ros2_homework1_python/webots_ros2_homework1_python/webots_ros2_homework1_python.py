import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math
import time

# Constants for the robot control
LINEAR_VEL = 0.22
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
LIDAR_AVOID_DISTANCE = 0.7
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR
RIGHT_SIDE_INDEX = 270
RIGHT_FRONT_INDEX = 210
LEFT_FRONT_INDEX = 150
LEFT_SIDE_INDEX = 90

# Stall detection constants
STALL_THRESHOLD = 0.05  # Movement threshold to detect a stall
STALL_TIME_THRESHOLD = 3  # Time in seconds before stall is detected
STARTUP_DELAY = 5  # Delay before checking for stalls
STALL_LIDAR_THRESHOLD = 0.1  # Minimum distance for stall detection with LiDAR

class HeuristicSearch(Node):

    def __init__(self):
        super().__init__('heuristic_search_node')
        
        # Robot state
        self.scan_cleaned = []
        self.stall_detected = False
        self.turtlebot_moving = False
        self.laser_forward = 0
        self.odom_data = 0
        self.cmd = Twist()
        self.timer_period = 0.5
        
        # Path tracking variables
        self.total_path_length = 0.0
        self.most_distant_point = 0.0
        self.start_position = None
        self.last_position = None
        self.path = []
        self.trial_logs = []  # Logs for each trial
        
        # Stall detection variables
        self.last_position_check = None
        self.stall_time = 0
        self.start_time = time.time()  # Start time for delay

        # ROS2 publishers and subscribers
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscriber1 = self.create_subscription(
            LaserScan, '/scan', self.listener_callback1, 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber2 = self.create_subscription(
            Odometry, '/odom', self.listener_callback2, 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def listener_callback1(self, msg1):
        scan = msg1.ranges
        self.scan_cleaned = []
        for reading in scan:
            if reading == float('Inf'):
                self.scan_cleaned.append(3.5)
            elif math.isnan(reading):
                self.scan_cleaned.append(0.0)
            else:
                self.scan_cleaned.append(reading)

    def listener_callback2(self, msg2):
        position = msg2.pose.pose.position
        if self.start_position is None:
            # Record the start position at the beginning of the trial
            self.start_position = position
        
        if self.last_position is not None:
            # Calculate distance moved since last position
            dx = position.x - self.last_position.x
            dy = position.y - self.last_position.y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Add to total path length
            self.total_path_length += distance
            
            # Calculate distance from start point
            start_distance = math.sqrt((position.x - self.start_position.x)**2 + 
                                       (position.y - self.start_position.y)**2)
            if start_distance > self.most_distant_point:
                self.most_distant_point = start_distance

        # Stall detection logic
        self.detect_stall(position)

        # Store the current position for path plotting
        self.path.append((position.x, position.y))
        self.last_position = position

    def detect_stall(self, current_position):
        # Wait for the startup delay before activating stall detection
        if time.time() - self.start_time < STARTUP_DELAY:
            return

        # If this is the first check, set the last_position_check
        if self.last_position_check is None:
            self.last_position_check = current_position
            return
        
        # Calculate the distance moved since last check
        dx = current_position.x - self.last_position_check.x
        dy = current_position.y - self.last_position_check.y
        movement = math.sqrt(dx**2 + dy**2)

        # Check LiDAR for obstacle very close to robot
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])
        is_obstacle_near = front_lidar_min < STALL_LIDAR_THRESHOLD

        if movement < STALL_THRESHOLD and is_obstacle_near:
            # Increment stall time if movement is below threshold and an obstacle is near
            self.stall_time += self.timer_period
            if self.stall_time >= STALL_TIME_THRESHOLD:
                self.stall_detected = True
                self.recover_from_stall()
        else:
            # Reset stall detection if movement is above threshold
            self.stall_time = 0
            self.stall_detected = False

        # Update the position check
        self.last_position_check = current_position

    def recover_from_stall(self):
        # Take corrective action to get unstuck
        self.get_logger().info('Stall detected, recovering...')
        self.cmd.linear.x = 0.0  # Stop moving forward
        self.cmd.angular.z = 0.5  # Rotate 90 degrees
        self.publisher_.publish(self.cmd)
        rclpy.spin_once(self, timeout_sec=1.0)  # Rotate for 1 second
        
        # Stop rotating and try moving forward again
        self.cmd.angular.z = 0.0
        self.cmd.linear.x = LINEAR_VEL
        self.publisher_.publish(self.cmd)
        rclpy.spin_once(self, timeout_sec=2.0)  # Move forward for 2 seconds

        # Reset the stall flag after recovery attempt
        self.stall_detected = False
        self.get_logger().info('Stall recovery attempted, moving again...')

    def timer_callback(self):
        if len(self.scan_cleaned) == 0:
            self.turtlebot_moving = False
            return
        
        left_lidar_min = min(self.scan_cleaned[LEFT_SIDE_INDEX:LEFT_FRONT_INDEX])
        right_lidar_min = min(self.scan_cleaned[RIGHT_FRONT_INDEX:RIGHT_SIDE_INDEX])
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])

        # Obstacle avoidance
        if front_lidar_min < SAFE_STOP_DISTANCE:
            if self.turtlebot_moving:
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.0
                self.publisher_.publish(self.cmd)
                self.turtlebot_moving = False
        elif front_lidar_min < LIDAR_AVOID_DISTANCE:
            self.cmd.linear.x = 0.07
            if right_lidar_min > left_lidar_min:
                self.cmd.angular.z = -0.3
            else:
                self.cmd.angular.z = 0.3
            self.publisher_.publish(self.cmd)
            self.turtlebot_moving = True
        else:
            self.cmd.linear.x = 0.3
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            self.turtlebot_moving = True

    def log_trial_data(self):
        # Save the path length and most distant point after each trial
        self.trial_logs.append({
            'total_path_length': self.total_path_length,
            'most_distant_point': self.most_distant_point,
            'path': self.path.copy()
        })
        # Reset for next trial
        self.total_path_length = 0.0
        self.most_distant_point = 0.0
        self.start_position = None
        self.last_position = None
        self.path = []

    def output_trial_logs(self):
        # Output the trial logs (you could also save this to a file)
        for idx, log in enumerate(self.trial_logs):
            print(f"Trial {idx+1}:")
            print(f"  Total Path Length: {log['total_path_length']:.2f} meters")
            print(f"  Most Distant Point: {log['most_distant_point']:.2f} meters")

    def plot_robot_path(self, trial_num):
        img = mpimg.imread('apartment.png')  # Replace with your actual image
        x_coords, y_coords = zip(*self.path)
        plt.imshow(img, extent=[0, 10, 0, 10])  # Adjust extent based on your image size
        plt.plot(x_coords, y_coords, label=f'Trial {trial_num}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Robot Path for Trial {trial_num}')
        plt.legend()
        plt.savefig(f'robot_path_trial_{trial_num}.png')
        plt.show()

    def plot_all_trials(self):
        img = mpimg.imread('apartment.png')  # Replace with your actual image
        plt.imshow(img, extent=[0, 10, 0, 10])  # Adjust extent based on your image size
        for trial_num, log in enumerate(self.trial_logs):
            path = log['path']
            x_coords, y_coords = zip(*path)
            plt.plot(x_coords, y_coords, label=f'Trial {trial_num + 1}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Paths for 5 Trials')
        plt.legend()
        plt.savefig('robot_paths_all_trials.png')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    heuristic_search_node = HeuristicSearch()
    rclpy.spin(heuristic_search_node)
    heuristic_search_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
