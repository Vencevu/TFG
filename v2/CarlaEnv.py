from cmath import inf
import carla
import math
import time
import numpy as np
import Config

class CarlaEnv:

    front_camera = None
    collision = None
    lidar_data = None
    getting_data = False
    vehicle = None
    obj_prox = 0
    distance_to_obstacle = 0

    def __init__(self):
        self.reward = Config.MIN_REWARD
        self.done = False

        self.actor_list = []
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    def reset(self):
        self.destroy_all_actors()
        self.gen_vehicle()
        time.sleep(0.5)
        self.add_sensor("rgb_cam")
        self.add_sensor("obs_det")
        self.collision = None
        self.add_sensor("col_det")
        self.obj_prox = 0
        self.add_sensor("lidar")
        self.episode_start = time.time()

    def destroy_all_actors(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def gen_vehicle(self):
        bp = self.blueprint_library.filter('vehicle')[0]
        transform = carla.Transform(carla.Location(Config.START_X, Config.START_Y, 0.6))
        self.vehicle = self.world.try_spawn_actor(bp, transform) 
        while(self.vehicle == None):
            transform.location.x -= 1
            self.vehicle = self.world.try_spawn_actor(bp, transform) 
        self.actor_list.append(self.vehicle)
        

    def add_sensor(self, sensor_name):
        if sensor_name == "rgb_cam":
            bp = self.blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(Config.IM_WIDTH))
            bp.set_attribute('image_size_y', str(Config.IM_HEIGHT))
            bp.set_attribute('fov', '110')
            bp.set_attribute('sensor_tick', '1.0')
            transform = carla.Transform(carla.Location(x=0.8, z=1.7))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_img(data))
        
        if sensor_name == "obs_det":
            bp = self.blueprint_library.find('sensor.other.obstacle')
            bp.set_attribute('sensor_tick', '1.0')

            transform = carla.Transform(carla.Location(x = 0.8, y = -0.5, z = 0.5))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_obs(data))

            transform = carla.Transform(carla.Location(x = 0.8, y = 0.5, z = 0.5))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_obs(data))
        
        if sensor_name == "col_det":
            bp = self.blueprint_library.find('sensor.other.collision')
            transform = carla.Transform(carla.Location(z=0.5))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_col(data))
        
        if sensor_name == "lidar":
            lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('upper_fov', str(15))
            lidar_bp.set_attribute('lower_fov', str(-30))
            lidar_bp.set_attribute('horizontal_fov', str(180.0))
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            lidar_bp.set_attribute('channels',str(64))
            lidar_bp.set_attribute('points_per_second',str(100000))
            lidar_bp.set_attribute('rotation_frequency',str(40))
            lidar_bp.set_attribute('range',str(20))

            lidar_location = carla.Location(0, 0, 1.5)
            lidar_transform = carla.Transform(lidar_location)
            lidar_sen = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to=self.vehicle)

            lidar_sen.listen(self.process_lidar)
            self.actor_list.append(lidar_sen)


    def vehicle_velocity(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return kmh
    
    def distance_to_goal(self, goal_pose_x, goal_pose_y):
        """Euclidean distance between current pose and the goal."""
        pose_x = self.vehicle.get_location().x
        pose_y = self.vehicle.get_location().y
        a = (goal_pose_x - pose_x)
        b = (goal_pose_y - pose_y)
        return math.sqrt(pow((a), 2) + pow((b), 2))

    def distance_to_detection(self, x, y, z):
        dist = math.sqrt(pow(0 - x, 2) + pow(0 - y, 2) + pow(-1 - z, 2))
        return dist

    def step(self, action, distance):
        self.done = False
        reset_type = 0

        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, brake=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=0.7))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.70, steer=0.5, brake=0))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.70, steer=-0.5, brake=0))
    
        vel = self.vehicle_velocity()

        # Penalizacion por velocidad
        if vel < 5 or vel > 20:
            self.reward = Config.MIN_REWARD
        else:
            self.reward = Config.INT_REWARD
        
        # Recompensa en funcion de cuanto se acerca al objetivo
        if distance > 1:
            self.reward += Config.INT_REWARD * (1/distance)
        else:
            self.reward += Config.MAX_REWARD
            self.done = True
            reset_type = 1
            print("Objetivo alcanzado")

        # Penalizacion por aproximacion a obstaculo
        if self.distance_to_obstacle > 0:
            self.reward += Config.MIN_REWARD / self.distance_to_obstacle
            self.distance_to_obstacle = 0
        
        # Penalizacion por aproximacion a obstaculo (LIDAR)
        if self.obj_prox > 0:
            self.reward = Config.INT_REWARD * self.obj_prox

        # Reinicio por colision
        if self.collision != None:
            self.reward = Config.MIN_REWARD * 3
            reset_type = 2
            self.done = True
            print("Collision-Reset...")
            

        # Reinicio por tiempo
        if self.episode_start + Config.SECONDS_PER_EPISODE < time.time():  
            self.done = True
            reset_type = 1
            self.reward += Config.MIN_REWARD
            print("Time-Reset...")
        
        return self.front_camera, vel, self.reward, self.done, reset_type

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((Config.IM_HEIGHT, Config.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        self.front_camera = i3

    def process_col(self, col):
        if self.collision is None:
            self.collision = col

    def process_lidar(self, point_cloud):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        # Isolate the 3D data
        points = data[:, :-1]
        # Eliminamos puntos pertenecientes al suelo y al propio coche
        points = points[points[:, 2] > -1, :]
        points = points[np.sqrt(points[:, 0]**2 + points[:, 1]**2) > 1.9, :]
        # Calculamos vector de distancias
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        D = np.sqrt(X**2 + Y**2 + (Z + 1)**2)
        # Nos quedamos con distancias menores a 4 metros
        D = D[D[:] < 4]

        if D.shape[0] > 0:
            self.obj_prox = np.amin(D)

    def process_obs(self, obs):
        if obs != None:
            self.distance_to_obstacle = obs.distance
        else:
            self.distance_to_obstacle = 0