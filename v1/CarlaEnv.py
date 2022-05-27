from cmath import inf
import carla
import math
import time
import numpy as np
from termcolor import colored
import Config

class CarlaEnv:

    front_camera = None
    collision = None
    vehicle = None
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
            transform = carla.Transform(carla.Location(x=0.8, z=1.5))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_img(data))
        
        if sensor_name == "obs_det":
            bp = self.blueprint_library.find('sensor.other.obstacle')
            bp.set_attribute('sensor_tick', '1.0')
            bp.set_attribute('distance', '3')

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
        if vel < 5 or vel > 30:
            self.reward = Config.MIN_REWARD
        else:
            self.reward = Config.INT_REWARD
        
        # Recompensa en funcion de cuanto se acerca al objetivo
        if distance > 1:
            self.reward += Config.INT_REWARD / distance
        else:
            self.reward += Config.MAX_REWARD
            self.done = True
            reset_type = 1
            print(colored("Objetivo alcanzado", 'yellow'))

        # Penalizacion por aproximacion a obstaculo
        if self.distance_to_obstacle > 0:
            self.reward += Config.MIN_REWARD / self.distance_to_obstacle
            self.distance_to_obstacle = 0

        # Reinicio por colision
        if self.collision != None:
            self.done = True
            reset_type = 2
            self.reward = Config.MIN_REWARD * 3
            print("Collision-Reset...")
            

        # Reinicio por tiempo
        if self.episode_start + Config.SECONDS_PER_EPISODE < time.time():  
            self.done = True
            reset_type = 3
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

    def process_obs(self, obs):
        if obs != None:
            self.distance_to_obstacle = obs.distance
        else:
            self.distance_to_obstacle = 0