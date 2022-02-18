from spade import agent, quit_spade
import carla
import random
import queue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

#Clase Agente
class DummyAgent(agent.Agent):
    async def setup(self):

        #Conexion con CARLA
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)

        #Configuracion del mundo
        world = client.get_world()

        #Creamos vehiculo
        actor_list = []
        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle'))
        transform = random.choice(world.get_map().get_spawn_points()) 
        vehicle = world.spawn_actor(bp, transform) 
        actor_list.append(vehicle)

        #Creamos sensor y lo acoplamos al vehiculo
        camera_depth = blueprint_library.find('sensor.camera.depth')
        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera_d = world.spawn_actor(camera_depth, camera_transform, attach_to=vehicle)
        image_queue_depth = queue.Queue()
        camera_d.listen(image_queue_depth.put)
        actor_list.append(camera_d)

        #Guardamos informacion del sensor
        image_depth = image_queue_depth.get()
        image_depth.save_to_disk("test_images/%06d_depth.png" %(image_depth.frame), carla.ColorConverter.LogarithmicDepth)
        #client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        

#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")
future = dummy.start()
#Esperamos a que termine el setup
future.result()

dummy.stop()
quit_spade()
