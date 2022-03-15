import carla

class CarlaEnv:

    def __init__(self):
        self.actor_list = []
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    def reset(self):
        pass

    def destroy_all_actors(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def gen_vehicle(self):
        bp = self.blueprint_library.filter('vehicle')[0]
        transform = self.map.get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, transform) 
        self.actor_list.append(self.vehicle)