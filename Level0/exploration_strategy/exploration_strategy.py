import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Odometry
from geometry_msgs.msg import Twist
import math
import random
from nav_msgs.msg import OccupancyGrid

class RRTExploration(Node):

    def __init__(self):
        super().__init__('rrt_exploration')
        
        # Subscriber for LiDAR data
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        
        # Subscriber for odometry data
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # Subscriber for map data from SLAM
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        self.current_map = None
        
        # Publisher for robot movement commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Placeholder for the RRT algorithm
        self.rrt_tree = []
        
        # Robot's current position and orientation
        self.robot_position = (0, 0)
        self.robot_orientation = 0  # in radians
        
        # Last target point
        self.last_target_point = None
        self.target_retries = 0
        
        # Timer to periodically generate and publish movement commands
        self.timer = self.create_timer(0.1, self.generate_and_publish_command)  # 10Hz

        # For SLAM integration
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Configurable parameters
        self.safety_distance = 0.5  # Minimum distance to obstacles for safety
        self.rrt_frequency = 1  # Run RRT every second


    def lidar_callback(self, msg):
        # Safety check
        if min(msg.ranges) < self.safety_distance:
            self.stop_robot()
            return
        self.points = self.convert_to_occupancy_grid(msg)  # Store the points as an instance variable
        frontiers = self.identify_frontiers(self.points)
        self.target = self.choose_target(frontiers)  # Only update the target here

    def odom_callback(self, msg):
        # Update robot's position and orientation
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        _, _, self.robot_orientation = RRTExploration.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def map_callback(self, msg):
        # Update the current map
        self.current_map = msg
        # Update the environment dimensions based on the map
        self.environment_width = msg.info.width * msg.info.resolution
        self.environment_height = msg.info.height * msg.info.resolution
        
        
    def stop_robot(self):
        cmd = Twist()  # Zero velocities
        self.publisher.publish(cmd)
    
    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z  # in radians
    
    def convert_to_occupancy_grid(self, msg):
        # For simplicity, let's assume the LiDAR has a 360-degree view.
        # We'll convert the LaserScan data into a list of (x, y) points representing detected obstacles.
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append((x, y))
            angle += msg.angle_increment
        # Here, you can convert these points into an occupancy grid if needed.
        # For now, we'll return the points directly.
        return points

    def is_frontier(self, x, y, grid):
        # Check if the cell is known
        if grid[y][x] == 'known':
            # Check the neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= x + dx < len(grid[0]) and 0 <= y + dy < len(grid) and grid[y + dy][x + dx] == 'unknown':
                        return True
        return False

    def points_to_grid(self, points):
        # Define the size of the grid based on the environment dimensions and resolution
        grid_width = 200  # Example value
        grid_height = 200  # Example value
        resolution = 0.05  # 5 cm per cell
    
        # Initialize the grid with 'unknown' values
        grid = [['unknown' for _ in range(grid_width)] for _ in range(grid_height)]
    
        # Convert each point to its grid coordinates and mark it as 'obstacle'
        for point in points:
            x, y = point
            grid_x = int(x / resolution)
            grid_y = int(y / resolution)
    
            # Ensure the coordinates are within the grid boundaries
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                grid[grid_y][grid_x] = 'obstacle'
    
        # You can further refine this by marking the robot's current position and its immediate surroundings as 'known'
        robot_grid_x = int(self.robot_position[0] / resolution)
        robot_grid_y = int(self.robot_position[1] / resolution)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= robot_grid_x + dx < grid_width and 0 <= robot_grid_y + dy < grid_height:
                    grid[robot_grid_y + dy][robot_grid_x + dx] = 'known'
    
        return grid


    def identify_frontiers(self, points):
        # Convert points to a grid representation
        grid = self.points_to_grid(points)
        
        frontiers = []
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if self.is_frontier(x, y, grid):
                    frontiers.append((x, y))
                    
        return frontiers

    
    def choose_target(self, frontiers):
        # Use the RRT algorithm to choose a target from the frontiers.
        target = self.rrt_algorithm()
        
        # Check if the robot is stuck trying to reach the same target
        if self.last_target_point == target:
            self.target_retries += 1
            if self.target_retries > 5:  # If the robot has been stuck for 5 cycles, choose a new target
                target = random.choice(frontiers)
                self.target_retries = 0
        else:
            self.target_retries = 0
        
        self.last_target_point = target
        return target


    def rrt_algorithm(self):
        # Only run the RRT algorithm based on the set frequency
        current_time = self.get_clock().now().seconds
        if current_time % self.rrt_frequency == 0:
            # Initialize the tree with the robot's current position
            tree = [self.robot_position]
            
            # Define the number of iterations
            num_iterations = 1000
            
            for _ in range(num_iterations):
                # Randomly sample a point in the environment
                sampled_point = self.sample_point()
                
                # Find the nearest node in the tree to the sampled point
                nearest_node = self.find_nearest(tree, sampled_point)
                
                # Steer towards the sampled point from the nearest node
                new_node = self.steer(nearest_node, sampled_point)
                
                # Check if the path between the nearest node and the new node is free of obstacles
                if self.is_path_free(nearest_node, new_node, self.points):  # Pass the points argument
                    # Add the new node to the tree
                    tree.append(new_node)
                    
                    # Optionally, check if the new node is close to a goal or target
                    if self.is_goal_reached(new_node):
                        return new_node
        
            # If the goal is not reached or the environment is not fully explored, 
            # you can return the last node added or any other strategy you prefer
            return tree[-1]
        else:
            return self.last_target_point  # Return the last target if not time to run RRT

    
    def sample_point(self):
        # Randomly sample a point in the environment based on the map dimensions.
        x = random.uniform(0, self.environment_width)
        y = random.uniform(0, self.environment_height)
        return (x, y)
    
    def find_nearest(self, tree, point):
        nearest_node = tree[0]
        min_distance = float('inf')
        for node in tree:
            distance = math.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node

    
    def steer(self, nearest_node, sampled_point):
        # Move a fixed distance towards the sampled point.
        # Assuming a step size of 0.5 meters for now.
        step_size = 0.5
        theta = math.atan2(sampled_point[1] - nearest_node[1], sampled_point[0] - nearest_node[0])
        new_x = nearest_node[0] + step_size * math.cos(theta)
        new_y = nearest_node[1] + step_size * math.sin(theta)
        return (new_x, new_y)

    
    def is_path_free(self, start, end, points):
        if points is None:
            points = self.convert_to_occupancy_grid(self.subscription)
        # Implement Bresenham's line algorithm to check if the path between start and end is free of obstacles
        x1, y1 = start
        x2, y2 = end
        obstacles = set(points)  # Convert list of points to a set for O(1) lookups
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if (x1, y1) in obstacles:
                return False
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return True

    def is_goal_reached(self, node):
        # Check if the node is close to a frontier or target
        # For this example, we'll assume a goal is reached if the node is within a certain distance of a frontier
        for frontier in self.frontiers:
            distance = math.sqrt((node[0] - frontier[0])**2 + (node[1] - frontier[1])**2)
            if distance < 0.5:  # Threshold can be adjusted
                return True
        return False
    
    def generate_movement_command(self, target_point):
        # Calculate the direction and distance to the target.
        theta = math.atan2(target_point[1] - self.robot_position[1], target_point[0] - self.robot_position[0])
        distance = math.sqrt((target_point[0] - self.robot_position[0])**2 + (target_point[1] - self.robot_position[1])**2)
        
        cmd = Twist()
        if distance > 0.5:  # If the target is more than 0.5 meters away, move forward.
            cmd.linear.x = 0.5
        # Simple control strategy to turn smoothly towards the target
        cmd.angular.z = 0.5 * (theta - self.robot_orientation)
        return cmd
    
    def generate_and_publish_command(self):
        # Error handling: Check if we have a valid target
        if not self.target:
            self.get_logger().warn("No valid target found. Stopping robot.")
            self.stop_robot()
            return

        # Check if the path to the target is blocked by an obstacle
        if not self.is_path_free(self.robot_position, self.target, self.points):
            self.get_logger().warn("Path to target is blocked. Replanning...")
            # Re-run the RRT algorithm to find a new path
            self.target = self.rrt_algorithm()

        cmd = self.generate_movement_command(self.target)
        self.publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RRTExploration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
