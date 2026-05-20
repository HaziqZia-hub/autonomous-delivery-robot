import math
import pygame
import time
import numpy as np
import heapq
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation
from gpiozero import Motor, RotaryEncoder
from mpu6050 import mpu6050
from rplidar import RPLidar, RPLidarException

# ---------------- CONSTANTS & HARDWARE ----------------
TICKS_PER_REV = 48
WHEEL_RADIUS_M = 0.032
WHEEL_CIRCUMFERENCE_M = 2 * math.pi * WHEEL_RADIUS_M
METERS_PER_TICK = WHEEL_CIRCUMFERENCE_M / TICKS_PER_REV

WHEEL_BASE_M = 0.17
  
left_motor = Motor(forward=4, backward=17)
right_motor = Motor(forward=22, backward=27)
right_encoder = RotaryEncoder(a=25, b=24, max_steps=1000000)
left_encoder = RotaryEncoder(a=12, b=16, max_steps=1000000)
sensor = mpu6050(0x68)

PORT_NAME = '/dev/ttyUSB0'
BAUD_RATE = 115200

# ---------------- SPEED & CALIBRATION ----------------
BASE_SPEED = 0.6
  
FWD_LEFT_TRIM = 0.9  
FWD_RIGHT_TRIM = 1.0  
BWD_LEFT_TRIM = 0.85  
BWD_RIGHT_TRIM = 1.0  

MIN_STALL_PWM = 0.35  

def true_power(requested_speed):
    requested_speed = max(0.0, min(1.0, requested_speed))
    if requested_speed < 0.05:
        return 0.0 
    return MIN_STALL_PWM + (requested_speed * (1.0 - MIN_STALL_PWM))

def backward(speed=BASE_SPEED):
    left_motor.backward(true_power(speed * BWD_LEFT_TRIM))
    right_motor.backward(true_power(speed * BWD_RIGHT_TRIM))

def left(speed=BASE_SPEED):
    left_motor.backward(true_power(speed * BWD_LEFT_TRIM))
    right_motor.forward(true_power(speed * FWD_RIGHT_TRIM))

def right(speed=BASE_SPEED):
    left_motor.forward(true_power(speed * FWD_LEFT_TRIM))
    right_motor.backward(true_power(speed * BWD_RIGHT_TRIM))

def stop():
    left_motor.stop()
    right_motor.stop()

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

    def reset(self):
        self.prev_error = 0
        self.integral = 0

# Base PID for manual driving stabilization
pid = PIDController(kp=0.05, ki=0.0, kd=0.01)

# NEW: Autonomous Steering PID (Eliminates the "snaking")
nav_pid = PIDController(kp=0.8, ki=0.0, kd=0.15)

# ---------------- A* PATHFINDING ALGORITHM (OPTIMIZED) ----------------
def a_star(start_grid, goal_grid, occ_grid, width, height):
    walls = (occ_grid == 100)
    inflated_walls = binary_dilation(walls, iterations=2) 
    
    sx, sy = start_grid
    inflated_walls[max(0, sx-1):min(width, sx+2), max(0, sy-1):min(height, sy+2)] = False
    
    if inflated_walls[goal_grid[0], goal_grid[1]]:
        print("Goal is blocked or too close to a wall!")
        return []

    neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {}
    g_score = {start_grid: 0}
    
    max_nodes = 20000 
    nodes_expanded = 0
    
    while open_set and nodes_expanded < max_nodes:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal_grid:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy

            if 0 <= nx < width and 0 <= ny < height:
                if inflated_walls[nx, ny]:
                    continue 

                cost = 1.414 if abs(dx) == 1 and abs(dy) == 1 else 1.0
                tentative_g = g_score[current] + cost

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    h = math.hypot(goal_grid[0] - nx, goal_grid[1] - ny)
                    heapq.heappush(open_set, (tentative_g + h, (nx, ny)))

    print("Pathfinding timeout or route unreachable.")
    return [] 

# ---------------- VOXEL GRID FILTER ----------------
def voxel_filter(points, leaf_size_m=0.05):
    if len(points) == 0:
        return points
    pts = np.array(points)
    grid_coords = np.round(pts / leaf_size_m).astype(int)
    _, indices = np.unique(grid_coords, axis=0, return_index=True)
    return pts[indices].tolist()

# ---------------- OCCUPANCY GRID ----------------
class OccupancyGrid:
    def __init__(self, width_m, height_m, resolution_m):
        self.resolution = resolution_m
        self.grid_width = int(width_m / resolution_m)
        self.grid_height = int(height_m / resolution_m)
        
        self.grid = np.full((self.grid_width, self.grid_height), -1, dtype=np.int8)
        
        self.origin_cx = self.grid_width // 2
        self.origin_cy = self.grid_height // 2

    def world_to_grid(self, x, y):
        cx = int(x / self.resolution) + self.origin_cx
        cy = int(y / self.resolution) + self.origin_cy
        return cx, cy

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append((x, y))
        return cells

    def update(self, robot_x, robot_y, global_lidar_points):
        rx, ry = self.world_to_grid(robot_x, robot_y)
        
        for pt_x, pt_y in global_lidar_points:
            wx, wy = self.world_to_grid(pt_x, pt_y)
            
            if 0 <= wx < self.grid_width and 0 <= wy < self.grid_height:
                line_cells = self.bresenham(rx, ry, wx, wy)
                for (cx, cy) in line_cells[:-1]:
                    if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height:
                        if self.grid[cx, cy] != 100: 
                            self.grid[cx, cy] = 0
                self.grid[wx, wy] = 100

# ---------------- ICP ALGORITHM ----------------
def icp_match(prev_points, curr_points, initial_guess=(0.0, 0.0, 0.0), max_iterations=15):
    if len(prev_points) < 10 or len(curr_points) < 10:
        return initial_guess

    dx, dy, dtheta = initial_guess
    
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                  [np.sin(dtheta),  np.cos(dtheta)]])
    t = np.array([dx, dy])

    src = np.dot(curr_points, R.T) + t
    dst = np.array(prev_points)

    for _ in range(max_iterations):
        tree = KDTree(dst)
        distances, indices = tree.query(src)

        valid = distances < 0.2  
        src_matched = src[valid]
        dst_matched = dst[indices[valid]]

        if len(src_matched) < 10:
            break

        centroid_src = np.mean(src_matched, axis=0)
        centroid_dst = np.mean(dst_matched, axis=0)

        src_centered = src_matched - centroid_src
        dst_centered = dst_matched - centroid_dst

        H = np.dot(src_centered.T, dst_centered)
        U, S, Vt = np.linalg.svd(H)
        R_opt = np.dot(Vt.T, U.T)

        if np.linalg.det(R_opt) < 0:
            Vt[1, :] *= -1
            R_opt = np.dot(Vt.T, U.T)

        t_opt = centroid_dst - np.dot(centroid_src, R_opt.T)
        src = np.dot(src, R_opt.T) + t_opt

    c_curr = np.mean(curr_points, axis=0)
    c_src = np.mean(src, axis=0)
    
    curr_centered = curr_points - c_curr
    src_centered = src - c_src
    
    H_final = np.dot(curr_centered.T, src_centered)
    U_f, S_f, Vt_f = np.linalg.svd(H_final)
    R_total = np.dot(Vt_f.T, U_f.T)
    
    if np.linalg.det(R_total) < 0:
        Vt_f[1, :] *= -1
        R_total = np.dot(Vt_f.T, U_f.T)
        
    net_dtheta = math.atan2(R_total[1, 0], R_total[0, 0])
    
    t_total = c_src - np.dot(c_curr, R_total.T)
    net_dx = t_total[0]
    net_dy = t_total[1]

    return net_dx, net_dy, net_dtheta

# ---------------- MAIN LOGGER LOOP ----------------
def run_logger():
    pygame.init()
    
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("UGV Live Mapping & Autonomous A*")
    font = pygame.font.SysFont(None, 24)
    
    map_surface = pygame.Surface((800, 800))     
    map_surface.fill((0, 0, 0))                  
    
    SCALE = 80 
    CENTER_X, CENTER_Y = 400, 400 
    
    occupancy_grid = OccupancyGrid(width_m=20.0, height_m=20.0, resolution_m=0.05)
    
    print("Connecting to LiDAR...")
    lidar = RPLidar(PORT_NAME, baudrate=BAUD_RATE, timeout=3)
    lidar.start_motor()
    time.sleep(2)

    log_file = open("slam_log.csv", "w")
    print("Recording to slam_log.csv...")

    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
    current_yaw, target_yaw = 0.0, 0.0
    
    odom_dx, odom_dy, odom_dtheta = 0.0, 0.0, 0.0
    prev_left_ticks, prev_right_ticks = 0, 0
    
    driving_forward = False                 
    driving_backward = False                
    
    last_motor_time = time.time()
    scan_distances = [0] * 360
    scans_recorded = 0
    
    global_map_points = []
    
    # NAVIGATION VARIABLES
    navigating = False
    active_path = []
    waypoint_index = 0
    stable_frames = 10 

    try:
        for new_scan, quality, angle, distance in lidar.iter_measures():
            current_time = time.time()

            # ==============================================================
            # 1. THE FAST LOOP (Odometry, Control, & Autonomy at 20Hz)
            # ==============================================================
            if current_time - last_motor_time >= 0.05:
                dt = current_time - last_motor_time
                last_motor_time = current_time

                gyro_data = sensor.get_gyro_data()
                raw_gyro_z = gyro_data['z']
                
                if abs(raw_gyro_z) < 1.5:
                    raw_gyro_z = 0.0
                
                gyro_z_rad_per_sec = math.radians(raw_gyro_z) 
                imu_dtheta = gyro_z_rad_per_sec * dt 
                current_yaw += raw_gyro_z * dt

                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        raise KeyboardInterrupt
                    
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        target_x = (mx - CENTER_X) / SCALE
                        target_y = (CENTER_Y - my) / SCALE

                        start_cx, start_cy = occupancy_grid.world_to_grid(robot_x, robot_y)
                        goal_cx, goal_cy = occupancy_grid.world_to_grid(target_x, target_y)

                        print("Calculating route...")
                        path_cells = a_star((start_cx, start_cy), (goal_cx, goal_cy), occupancy_grid.grid, occupancy_grid.grid_width, occupancy_grid.grid_height)

                        if path_cells:
                            active_path = []
                            for cx, cy in path_cells:
                                wx = (cx - occupancy_grid.origin_cx) * occupancy_grid.resolution
                                wy = (cy - occupancy_grid.origin_cy) * occupancy_grid.resolution
                                active_path.append((wx, wy))
                            
                            navigating = True
                            waypoint_index = 0
                            nav_pid.reset() # Reset steering PID for a fresh start
                            print(f"Path locked! {len(active_path)} waypoints to destination.")
                        else:
                            print("Navigation Failed.")
                    
                    elif event.type == pygame.KEYDOWN:
                        navigating = False 
                        stop()
                        if event.key == pygame.K_UP:
                            target_yaw = current_yaw
                            pid.reset()
                            driving_forward = True   
                            driving_backward = False 
                        elif event.key == pygame.K_DOWN:
                            target_yaw = current_yaw
                            pid.reset()
                            driving_backward = True  
                            driving_forward = False  
                        elif event.key == pygame.K_LEFT:
                            driving_forward, driving_backward = False, False
                            left()
                        elif event.key == pygame.K_RIGHT:
                            driving_forward, driving_backward = False, False
                            right()
                        elif event.key == pygame.K_SPACE:
                            driving_forward, driving_backward = False, False
                            stop()

                    elif event.type == pygame.KEYUP:
                        if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                            driving_forward, driving_backward = False, False
                            stop()
                         
                # -- MANUAL DRIVING --
                if driving_forward:
                    adjustment = pid.compute(target_yaw, current_yaw, dt)
                    raw_left = (BASE_SPEED * FWD_LEFT_TRIM) - adjustment
                    raw_right = (BASE_SPEED * FWD_RIGHT_TRIM) + adjustment
                    left_motor.forward(true_power(raw_left))
                    right_motor.forward(true_power(raw_right))
                elif driving_backward:
                    adjustment = pid.compute(target_yaw, current_yaw, dt)
                    raw_left = (BASE_SPEED * BWD_LEFT_TRIM) + adjustment 
                    raw_right = (BASE_SPEED * BWD_RIGHT_TRIM) - adjustment
                    left_motor.backward(true_power(raw_left))
                    right_motor.backward(true_power(raw_right))

                # -- AUTONOMOUS PATH FOLLOWER (PURE PURSUIT + PID) --
                elif navigating and len(active_path) > 0:
                    
                    # 1. Find the closest point on the path
                    min_dist = float('inf')
                    closest_idx = waypoint_index
                    search_window = min(waypoint_index + 30, len(active_path))
                    
                    for i in range(waypoint_index, search_window):
                        wx, wy = active_path[i]
                        dist = math.hypot(wx - robot_x, wy - robot_y)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i

                    waypoint_index = closest_idx

                    # 2. Check if destination reached
                    final_wx, final_wy = active_path[-1]
                    dist_to_goal = math.hypot(final_wx - robot_x, final_wy - robot_y)

                    if dist_to_goal < 0.25: 
                        navigating = False
                        stop()
                        print("Destination Reached!")
                    else:
                        # 3. PURE PURSUIT Lookahead
                        LOOKAHEAD_STEPS = 8 
                        target_idx = min(waypoint_index + LOOKAHEAD_STEPS, len(active_path) - 1)
                        target_wx, target_wy = active_path[target_idx]

                        dx = target_wx - robot_x
                        dy = target_wy - robot_y
                        target_angle = math.atan2(dy, dx)
                        angle_diff = (target_angle - robot_theta + math.pi) % (2 * math.pi) - math.pi

                        # 4. Steering Execution
                        if abs(angle_diff) > 0.4: 
                            # Extreme turn needed, spin in place
                            if angle_diff > 0:
                                left_motor.backward(true_power(0.4))
                                right_motor.forward(true_power(0.4))
                            else:
                                left_motor.forward(true_power(0.4))
                                right_motor.backward(true_power(0.4))
                        else:
                            # Smooth PID Steering
                            steer_adjust = nav_pid.compute(0, -angle_diff, dt)
                            
                            # Dynamic speed reduction on sharper curves
                            speed_multiplier = max(0.4, 1.0 - abs(angle_diff))
                            fwd_speed = BASE_SPEED * speed_multiplier

                            raw_left = (fwd_speed * FWD_LEFT_TRIM) - steer_adjust
                            raw_right = (fwd_speed * FWD_RIGHT_TRIM) + steer_adjust
                            
                            left_motor.forward(true_power(raw_left))
                            right_motor.forward(true_power(raw_right))

                # -- READ ENCODERS & SENSOR FUSION --
                current_left_ticks = left_encoder.steps
                current_right_ticks = right_encoder.steps
                
                d_left = (current_left_ticks - prev_left_ticks) * METERS_PER_TICK
                d_right = (current_right_ticks - prev_right_ticks) * METERS_PER_TICK
                
                d_center = (d_left + d_right) / 2.0 
                encoder_dtheta = (d_right - d_left) / WHEEL_BASE_M
                
                ALPHA = 0.85 
                fused_dtheta = (ALPHA * imu_dtheta) + ((1.0 - ALPHA) * encoder_dtheta)

                odom_dx += d_center * math.cos(odom_dtheta)
                odom_dy += d_center * math.sin(odom_dtheta)
                odom_dtheta += fused_dtheta 
                
                prev_left_ticks = current_left_ticks
                prev_right_ticks = current_right_ticks

            # ==============================================================
            # 2. THE SLOW LOOP (Scan-to-Map SLAM)
            # ==============================================================
            if quality > 10 and 200 <= distance <= 3000:
                angle_idx = min(359, int(angle))
                scan_distances[angle_idx] = distance / 1000.0

            if new_scan:
                curr_scan_local = []
                for angle_deg, dist_m in enumerate(scan_distances):
                    if dist_m > 0:
                        adjusted_angle = (angle_deg + 180) % 360
                        angle_rad = math.radians((360.0 - adjusted_angle) % 360)
                        lx = dist_m * math.cos(angle_rad)
                        ly = dist_m * math.sin(angle_rad)
                        curr_scan_local.append([lx, ly])

                curr_scan_local = voxel_filter(curr_scan_local, leaf_size_m=0.05)

                turn_speed = abs(math.degrees(odom_dtheta))
                if turn_speed > 0.5:
                    stable_frames = 0
                else:
                    stable_frames += 1
                
                is_safe_to_map = stable_frames >= 3

                robot_theta += odom_dtheta
                robot_x += odom_dx * math.cos(robot_theta) - odom_dy * math.sin(robot_theta)
                robot_y += odom_dx * math.sin(robot_theta) + odom_dy * math.cos(robot_theta)

                global_hits = []

                if len(global_map_points) > 50 and len(curr_scan_local) > 10:
                    
                    if is_safe_to_map:
                        local_map_target = []
                        for gx, gy in global_map_points:
                            if abs(gx - robot_x) < 3.0 and abs(gy - robot_y) < 3.0:
                                dx_p = gx - robot_x
                                dy_p = gy - robot_y
                                lx = dx_p * math.cos(robot_theta) + dy_p * math.sin(robot_theta)
                                ly = -dx_p * math.sin(robot_theta) + dy_p * math.cos(robot_theta)
                                local_map_target.append([lx, ly])
                        
                        if len(local_map_target) > 20:
                            if len(local_map_target) > 500:
                                local_map_target = np.random.permutation(local_map_target)[:500].tolist()

                            corr_dx, corr_dy, corr_dtheta = icp_match(
                                local_map_target, 
                                np.array(curr_scan_local), 
                                (0.0, 0.0, 0.0) 
                            )

                            # --- ICP CLAMPING TO PREVENT MAP SNAPPING ---
                            MAX_SHIFT = 0.05   
                            MAX_ANGLE = 0.035  
                            
                            corr_dx = max(-MAX_SHIFT, min(MAX_SHIFT, corr_dx))
                            corr_dy = max(-MAX_SHIFT, min(MAX_SHIFT, corr_dy))
                            corr_dtheta = max(-MAX_ANGLE, min(MAX_ANGLE, corr_dtheta))

                            robot_theta += corr_dtheta
                            robot_x += corr_dx * math.cos(robot_theta) - corr_dy * math.sin(robot_theta)
                            robot_y += corr_dx * math.sin(robot_theta) + corr_dy * math.cos(robot_theta)

                    for lx, ly in curr_scan_local:
                        gx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
                        gy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))
                        global_hits.append([gx, gy])

                    if is_safe_to_map:
                        occupancy_grid.update(robot_x, robot_y, global_hits)
                        global_map_points.extend(global_hits)
                        
                        global_map_points = voxel_filter(global_map_points, leaf_size_m=0.1)
                        if len(global_map_points) > 3000:
                            global_map_points = global_map_points[-3000:]
                        
                        for pt_x, pt_y in global_hits:
                            pt_px = int(CENTER_X + (pt_x * SCALE))
                            pt_py = int(CENTER_Y - (pt_y * SCALE))
                            if 0 <= pt_px < 800 and 0 <= pt_py < 800:
                                map_surface.set_at((pt_px, pt_py), (255, 0, 0))
                
                elif len(global_map_points) <= 50:
                    for lx, ly in curr_scan_local:
                        gx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
                        gy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))
                        global_hits.append([gx, gy])
                    occupancy_grid.update(robot_x, robot_y, global_hits)
                    global_map_points.extend(global_hits)

                odom_dx, odom_dy, odom_dtheta = 0.0, 0.0, 0.0

                # ---> DYNAMIC VISUALS <---
                screen.fill((0, 0, 0))             
                screen.blit(map_surface, (0, 0))   
                
                if navigating and len(active_path) > 0:
                    for i in range(waypoint_index, len(active_path) - 1):
                        p1_x = int(CENTER_X + (active_path[i][0] * SCALE))
                        p1_y = int(CENTER_Y - (active_path[i][1] * SCALE))
                        p2_x = int(CENTER_X + (active_path[i+1][0] * SCALE))
                        p2_y = int(CENTER_Y - (active_path[i+1][1] * SCALE))
                        pygame.draw.line(screen, (0, 0, 255), (p1_x, p1_y), (p2_x, p2_y), 2)

                for pt_x, pt_y in global_hits:
                    pt_px = int(CENTER_X + (pt_x * SCALE))
                    pt_py = int(CENTER_Y - (pt_y * SCALE))
                    if 0 <= pt_px < 800 and 0 <= pt_py < 800:
                        pygame.draw.circle(screen, (255, 165, 0), (pt_px, pt_py), 1)

                robot_px = int(CENTER_X + (robot_x * SCALE))
                robot_py = int(CENTER_Y - (robot_y * SCALE))
                
                end_x = robot_px + int(15 * math.cos(robot_theta))
                end_y = robot_py - int(15 * math.sin(robot_theta))
                pygame.draw.line(screen, (0, 255, 255), (robot_px, robot_py), (end_x, end_y), 2)
                pygame.draw.circle(screen, (0, 255, 0), (robot_px, robot_py), 5)
                
                status_str = "Auto Navigating" if navigating else ("Mapping" if is_safe_to_map else "Waiting for Scan...")
                img = font.render(f"Scans: {scans_recorded} | Status: {status_str}", True, (255, 255, 255))
                screen.blit(img, (20, 20))
                pygame.display.flip()

                log_distances = [int(d * 100) if d > 0 else 0 for d in scan_distances]
                csv_line = f"{robot_x:.4f},{robot_y:.4f},{robot_theta:.4f}," + ",".join(map(str, log_distances)) + "\n"
                log_file.write(csv_line)
                scans_recorded += 1

                scan_distances = [0] * 360 

    except KeyboardInterrupt:
        print("\nStopping and saving map data...")
    finally:
        stop()
        log_file.close()
        np.save("occupancy_map.npy", occupancy_grid.grid)
        lidar.stop()
        lidar.disconnect()
        pygame.quit()
        print(f"Data Collection Complete. Map saved to occupancy_map.npy!")

if __name__ == "__main__":
    run_logger()