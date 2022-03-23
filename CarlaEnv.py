import carla
import math
import time
import numpy as np
import Config

class CarlaEnv:

    front_camera = None

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
        time.sleep(0.5)
        self.gen_vehicle()
        time.sleep(0.5)
        self.add_sensor("rgb_cam")
        self.episode_start = time.time()

    def destroy_all_actors(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def gen_vehicle(self):
        bp = self.blueprint_library.filter('vehicle')[0]
        transform = self.map.get_spawn_points()[0]
        self.vehicle = self.world.try_spawn_actor(bp, transform) 
        self.actor_list.append(self.vehicle)

    def add_sensor(self, sensor):
        if sensor == "rgb_cam":
            bp = self.blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(Config.IM_WIDTH))
            bp.set_attribute('image_size_y', str(Config.IM_HEIGHT))
            bp.set_attribute('fov', '110')
            bp.set_attribute('sensor_tick', '1.0')
            transform = carla.Transform(carla.Location(x=0.8, z=1.7))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)
            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_img(data))

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
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.50, brake=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.50))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.00))
    
        vel = self.vehicle_velocity()

        if(vel < 10 or vel > 30):
            self.reward = Config.MIN_REWARD
            self.done = False
        else:
            self.reward = Config.INT_REWARD
            self.done = False
        
        if distance > 7:
            self.reward += Config.MIN_REWARD * 3
            self.done = False
        elif distance > 2:
            self.reward += Config.MIN_REWARD
            self.done = False
        else:
            self.reward += Config.MAX_REWARD
            self.done = True
            print("Objetivo alcanzado")
        
        if self.episode_start + Config.SECONDS_PER_EPISODE < time.time():  ## when to stop
            self.done = True
            print("Time-Reset...")
            print("Distancia a objetivo: ", distance)
        
        return self.front_camera, vel, self.reward, self.done, None

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((Config.IM_HEIGHT, Config.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        self.front_camera = i3