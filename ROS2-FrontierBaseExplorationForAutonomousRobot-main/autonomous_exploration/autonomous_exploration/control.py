import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid , Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq , math , random , yaml
import scipy.interpolate as si
import sys , threading , time

# Parameters
lookahead_distance = 0.24
speed = 0.18 # Max speed of the robot
expansion_size = 3 # How much the robot expands obstacles
target_error = 0.15 # Acceptable error margin when reaching target
robot_r = 0.2 # Safety distance/radius around the robot


pathGlobal = 0

# Convert quaternion to euler angles
def euler_from_quaternion(x,y,z,w):    
    # Calculate the yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    # Return the yaw angle (z-axis rotation)
    return yaw_z

# Calculate heuristic distance between two points
def heuristic(a, b):
    # Calculate and return the Euclidean distance between the two points using the Pythagorean theorem
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


# A* pathfinding algorithm
def astar(array, start, goal):
    # Define the possible neighbors for a given node (8 directions: up, down, left, right, and diagonals)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    
    # Set of nodes that have already been evaluated
    close_set = set()
    
    # Dictionary to keep track of the path
    came_from = {}
    
    # Cost from start to a node
    gscore = {start:0}
    
    # Estimated cost from start to goal through a node
    fscore = {start:heuristic(start, goal)}
    
    # Priority queue to keep nodes to be evaluated
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    # While there are nodes to be evaluated
    while oheap:
        # Get the node with the lowest fscore value
        current = heapq.heappop(oheap)[1]
        
        # If the current node is the goal, reconstruct and return the path
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        
        # Mark the current node as evaluated
        close_set.add(current)
        
        # Evaluate all neighbors of the current node
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            # Check if neighbor is within the bounds of the array
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    # Skip the neighbor if it's an obstacle
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # Skip if neighbor is outside y bounds
                    continue
            else:
                # Skip if neighbor is outside x bounds
                continue
            
            # If the neighbor has been evaluated and the new gscore is not better, skip it
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            
            # If this is a new path to the neighbor or a better one, update the path
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    # If no path to the goal was found, find the closest node to the goal and return its path
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    
    # If no path is found, return False
    return False


# B-spline path smoothing
def bspline_planning(array, sn):
    try:
        # Convert the input array to a numpy array
        array = np.array(array)
        
        # Extract the x and y coordinates from the array
        x = array[:, 0]
        y = array[:, 1]
        
        # Set the degree of the spline
        N = 2
        
        # Create a range for the length of x (used for spline fitting)
        t = range(len(x))
        
        # Compute B-spline representation of x and y
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        # Convert the tuple representation to a list for x
        x_list = list(x_tup)
        xl = x.tolist()
        # Append zeros to the coefficients (used for spline evaluation)
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        # Convert the tuple representation to a list for y
        y_list = list(y_tup)
        yl = y.tolist()
        # Append zeros to the coefficients (used for spline evaluation)
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        # Create a linear space for interpolation
        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        
        # Evaluate the spline for the interpolated values
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        
        # Combine the x and y coordinates to form the path
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        # If any error occurs, return the original array as the path
        path = array
    return path


# Pure pursuit path tracking algorithm
def pure_pursuit(current_x, current_y, current_heading, path, index):
    # Define the global variable for lookahead distance
    global lookahead_distance
    
    # Initialize the closest point to None
    closest_point = None
    
    # Set the initial speed
    v = speed
    
    # Iterate over the path starting from the given index
    for i in range(index, len(path)):
        x = path[i][0]
        y = path[i][1]
        
        # Calculate the distance between the current position and the path point
        distance = math.hypot(current_x - x, current_y - y)
        
        # If the distance is greater than the lookahead distance, set the closest point
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
            
    # If a closest point is found within the lookahead distance
    if closest_point is not None:
        # Calculate the heading towards the closest point
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        # Calculate the desired steering angle based on the difference between the target and current headings
        desired_steering_angle = target_heading - current_heading
    else:
        # If no point is found within the lookahead distance, target the last point in the path
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path) - 1
        
    # Normalize the steering angle to be between -pi and pi
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
        
    # Limit the steering angle to be within +/- pi/6 radians
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        # Set speed to zero if the steering angle is too large
        v = 0.0
        
    return v, desired_steering_angle, index


# Identify frontier cells in the occupancy grid
def frontierB(matrix):
    # Iterate over each row of the matrix
    for i in range(len(matrix)):
        # Iterate over each column of the matrix
        for j in range(len(matrix[i])):
            # Check if the current cell has a value of 0.0 (indicating unexplored)
            if matrix[i][j] == 0.0:
                # Check the cell above the current cell
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                # Check the cell below the current cell
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                # Check the cell to the left of the current cell
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                # Check the cell to the right of the current cell
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    # Return the updated matrix
    return matrix


# Assign groups to frontier cells
def assign_groups(matrix):
    # Initialize the group number
    group = 1
    # Create a dictionary to store the cells belonging to each group
    groups = {}
    
    # Iterate over each row of the matrix
    for i in range(len(matrix)):
        # Iterate over each column of the matrix
        for j in range(len(matrix[0])):
            # Check if the current cell is a frontier cell (value of 2)
            if matrix[i][j] == 2:
                # Use Depth First Search (DFS) to assign group numbers to connected frontier cells
                group = dfs(matrix, i, j, group, groups)
    
    # Return the updated matrix and the groups dictionary
    return matrix, groups


# Depth First Search (DFS) function to traverse and group connected frontier cells
def dfs(matrix, i, j, group, groups):
    # Check if the current cell is out of the matrix boundaries
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    
    # Check if the current cell is not a frontier cell (value of 2)
    if matrix[i][j] != 2:
        return group
    
    # If the current group number already exists in the groups dictionary, append the current cell to it
    if group in groups:
        groups[group].append((i, j))
    # If the current group number doesn't exist, create a new entry for it with the current cell
    else:
        groups[group] = [(i, j)]
    
    # Mark the current cell as visited by setting its value to 0
    matrix[i][j] = 0
    
    # Recursively call the dfs function for all neighboring cells (including diagonals)
    dfs(matrix, i + 1, j, group, groups)       # Below
    dfs(matrix, i - 1, j, group, groups)       # Above
    dfs(matrix, i, j + 1, group, groups)       # Right
    dfs(matrix, i, j - 1, group, groups)       # Left
    dfs(matrix, i + 1, j + 1, group, groups)   # Bottom right diagonal
    dfs(matrix, i - 1, j - 1, group, groups)   # Top left diagonal
    dfs(matrix, i - 1, j + 1, group, groups)   # Top right diagonal
    dfs(matrix, i + 1, j - 1, group, groups)   # Bottom left diagonal
    
    # Increment the group number and return it
    return group + 1


# Function to identify the top frontier groups based on their size
def fGroups(groups):
    # Sort the groups dictionary by the size of each group in descending order
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Extract the top five groups from the sorted list, but only include those groups that have more than 2 cells
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]
    
    # Return the top five groups
    return top_five_groups


# Calculate centroid of a set of points
def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

# This function is used to find the closest frontier group to the robot.
def findClosestGroup(matrix, groups, current, resolution, originX, originY):
    # Initialize the target point to None.
    targetP = None
    
    # Lists to store distances, paths, and scores.
    distances = []
    paths = []
    score = []
    
    # Initialize the maximum score index to -1.
    max_score = -1
    
    # Loop through each group to calculate the distance from the current position to the centroid of the group.
    for i in range(len(groups)):
        # Calculate the centroid of the group.
        middle = calculate_centroid([p[0] for p in groups[i][1]], [p[1] for p in groups[i][1]])
        
        # Find the shortest path from the current position to the centroid using the A* algorithm.
        path = astar(matrix, current, middle)
        
        # Convert the path coordinates using the given resolution and origin.
        path = [(p[1]*resolution+originX, p[0]*resolution+originY) for p in path]
        
        # Calculate the total length of the path.
        total_distance = pathLength(path)
        
        # Store the distance and path for later use.
        distances.append(total_distance)
        paths.append(path)
    
    # Calculate the score for each group based on its size and distance from the current position.
    for i in range(len(distances)):
        if distances[i] == 0:
            score.append(0)
        else:
            score.append(len(groups[i][1]) / distances[i])
    
    # Find the group with the highest score that is also sufficiently far from the current position.
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            if max_score == -1 or score[i] > score[max_score]:
                max_score = i
    
    # If a group with a high score is found, set its path as the target path.
    if max_score != -1:
        targetP = paths[max_score]
    else:
        # If no group meets the criteria, select a random group and a random point within that group as the target.
        # This helps the robot to escape from certain situations.
        index = random.randint(0, len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0, len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX, p[0]*resolution+originY) for p in path]
    
    # Return the target path.
    return targetP

# This function calculates the total length of a given path.
def pathLength(path):
    # Convert each point in the path to a tuple.
    for i in range(len(path)):
        path[i] = (path[i][0], path[i][1])
    
    # Convert the path to a numpy array for easier calculations.
    points = np.array(path)
    
    # Calculate the differences between consecutive points in the path.
    differences = np.diff(points, axis=0)
    
    # Calculate the Euclidean distance between consecutive points.
    distances = np.hypot(differences[:,0], differences[:,1])
    
    # Sum up all the distances to get the total length of the path.
    total_distance = np.sum(distances)
    
    return total_distance

# This function expands the obstacles in an occupancy grid by a given size.
def costmap(data, width, height, resolution):
    # Reshape the data into a 2D numpy array.
    data = np.array(data).reshape(height, width)
    
    # Find the coordinates of all the obstacles in the grid.
    wall = np.where(data == 100)
    
    # Loop through a square region defined by the expansion size around each obstacle.
    for i in range(-expansion_size, expansion_size+1):
        for j in range(-expansion_size, expansion_size+1):
            # Skip the center of the square (the obstacle itself).
            if i == 0 and j == 0:
                continue
            
            # Calculate the expanded coordinates.
            x = wall[0] + i
            y = wall[1] + j
            
            # Clip the coordinates to ensure they are within the grid boundaries.
            x = np.clip(x, 0, height-1)
            y = np.clip(y, 0, width-1)
            
            # Set the expanded coordinates to be obstacles.
            data[x, y] = 100
    
    # Multiply the data by the resolution to scale the values.
    data = data * resolution
    
    return data


# This is the main exploration function that guides the robot through an environment.
def exploration(data, width, height, resolution, column, row, originX, originY):
    global pathGlobal  # Global variable to store the path
    
    # Expand the obstacles in the occupancy grid for safer navigation.
    data = costmap(data, width, height, resolution)
    
    # Mark the robot's current position.
    data[row][column] = 0
    
    # Mark all cells with values greater than 5 as obstacles.
    data[data > 5] = 1
    
    # Identify frontier cells in the occupancy grid.
    data = frontierB(data)
    
    # Group the frontier cells.
    data, groups = assign_groups(data)
    
    # Sort the groups and select the largest 5 groups.
    groups = fGroups(groups)
    
    # If there are no groups, exploration is complete.
    if len(groups) == 0:
        path = -1
    else:
        # Mark unknown cells as non-traversable.
        data[data < 0] = 1
        
        # Find the closest frontier group to the robot.
        path = findClosestGroup(data, groups, (row, column), resolution, originX, originY)
        
        # If a path is found, smooth it using B-spline.
        if path != None:
            path = bspline_planning(path, len(path)*5)
        else:
            path = -1
    
    # Update the global path.
    pathGlobal = path
    return

# This function provides local control to the robot based on laser scan data.
def localControl(scan):
    v = None  # Linear velocity
    w = None  # Angular velocity
    
    # Check the first 60 scan points for obstacles.
    for i in range(60):
        if scan[i] < robot_r:
            v = 0.2
            w = -math.pi/4  # Turn left
            break
    
    # If no obstacles are detected in the first 60 points, check the last 60 points.
    if v == None:
        for i in range(300, 360):
            if scan[i] < robot_r:
                v = 0.2
                w = math.pi/4  # Turn right
                break
    
    return v, w


# This class provides the main node for navigation control.
class navigationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        
        # Create subscriptions to receive data from various topics.
        self.subscription = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # Create a publisher to send velocity commands.
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        print("[INFO] EXPLORATION MODE ACTIVE")
        self.discovered = True
        
        # Start the exploration function in a separate thread.
        threading.Thread(target=self.exp).start()
    
    # This function handles the exploration process.
    def exp(self):
        twist = Twist()
        while True:
            # Wait until sensor data is available.
            if not hasattr(self, 'map_data') or not hasattr(self, 'odom_data') or not hasattr(self, 'scan_data'):
                time.sleep(0.1)
                continue
            
            # If in exploration mode.
            if self.discovered == True:
                # If no path is available, start the exploration process.
                if isinstance(pathGlobal, int) and pathGlobal == 0:
                    column = int((self.x - self.originX)/self.resolution)
                    row = int((self.y - self.originY)/self.resolution)
                    exploration(self.data, self.width, self.height, self.resolution, column, row, self.originX, self.originY)
                    self.path = pathGlobal
                else:
                    self.path = pathGlobal
                
                # If exploration is complete.
                if isinstance(self.path, int) and self.path == -1:
                    print("[INFO] EXPLORATION COMPLETED")
                    sys.exit()
                
                # Set the target position.
                self.c = int((self.path[-1][0] - self.originX)/self.resolution)
                self.r = int((self.path[-1][1] - self.originY)/self.resolution)
                
                # Switch to path following mode.
                self.discovered = False
                self.i = 0
                print("[INFO] NEW TARGET SET")
                
                # Calculate the time to reach the target.
                t = pathLength(self.path)/speed
                t = t - 0.2
                self.t = threading.Timer(t, self.target_callback)
                self.t.start()
            
            # Path following block.
            else:
                # Get the robot's control commands.
                v, w = localControl(self.scan)
                if v == None:
                    v, w, self.i = pure_pursuit(self.x, self.y, self.yaw, self.path, self.i)
                
                # If the robot reaches the target.
                if abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error:
                    v = 0.0
                    w = 0.0
                    self.discovered = True
                    print("[INFO] TARGET REACHED")
                    self.t.join()
                
                # Publish the velocity commands.
                twist.linear.x = v
                twist.angular.z = w
                self.publisher.publish(twist)
                time.sleep(0.1)
    
    # Callback to restart the exploration process when close to the target.
    def target_callback(self):
        exploration(self.data, self.width, self.height, self.resolution, self.c, self.r, self.originX, self.originY)
    
    # Callback to handle laser scan data.
    def scan_callback(self, msg):
        self.scan_data = msg
        self.scan = msg.ranges
    
    # Callback to handle map data.
    def map_callback(self, msg):
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data
    
    # Callback to handle odometry data.
    def odom_callback(self, msg):
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

# Main function to initialize and run the node.
def main(args=None):
    rclpy.init(args=args)
    navigation_control = navigationControl()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

