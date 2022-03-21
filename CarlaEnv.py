import carla
import math
import Config

class CarlaEnv:

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

    def destroy_all_actors(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def gen_vehicle(self):
        bp = self.blueprint_library.filter('vehicle')[0]
        transform = self.map.get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, transform) 
        self.actor_list.append(self.vehicle)

    def vehicle_velocity(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return kmh

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.50, brake=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.50))
    
        vel = self.vehicle_velocity()
        if(vel < 10 or vel > 30):
            self.reward = Config.MIN_REWARD
        else:
            self.reward = Config.INT_REWARD
        
        return self.reward, self.done, None