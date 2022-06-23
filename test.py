import carla
import time

from matplotlib.colors import colorConverter

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(2.0)

world = client.get_world()
map = world.get_map()
blueprint_library = world.get_blueprint_library()
actor_list = []

bp = blueprint_library.filter('vehicle')[0]
transform = carla.Transform(carla.Location(-30, 26, 0.6))
vehicle = world.try_spawn_actor(bp, transform) 
actor_list.append(vehicle)

while True:
    try:
        time.sleep(2)
    except KeyboardInterrupt:
        break

client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
print("End")