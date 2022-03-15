import carla
import asyncio
import random
import time

class CarlaEnv:

    def __init__(self):
        self.actor_list = []
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    def reset(self):
        pass

    def gen_vehicle(self):
        bp = random.choice(self.blueprint_library.filter('vehicle'))
        transform = self.map.get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, transform) 
        self.actor_list.append(self.vehicle)