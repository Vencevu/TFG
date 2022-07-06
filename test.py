import carla
import time

actor_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(2.0)

world = client.get_world()
map = world.get_map()

blueprint_library = world.get_blueprint_library()
bp = blueprint_library.filter('vehicle')[0]
transform = carla.Transform(carla.Location(-30, 25, 0.6))
vehicle = world.spawn_actor(bp, transform) 
actor_list.append(vehicle)

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        break

client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
print("End...")