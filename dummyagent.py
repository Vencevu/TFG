from spade import agent, quit_spade
import carla
import random
import queue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

#Clase Agente
class DummyAgent(agent.Agent):
    
    actor_list = []
    world = None
    client = None

    async def setup(self):
        #Conexion con CARLA
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)

        #Configuracion del mundo
        self.world = self.client.get_world()

        #Creamos vehiculo
        
        blueprint_library = self.world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle'))
        transform = random.choice(self.world.get_map().get_spawn_points()) 
        vehicle = self.world.spawn_actor(bp, transform) 
        self.actor_list.append(vehicle)

        #Creamos sensor y lo acoplamos al vehiculo
        camera_depth = blueprint_library.find('sensor.camera.depth')
        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera_d = self.world.spawn_actor(camera_depth, camera_transform, attach_to=vehicle)
        image_queue_depth = queue.Queue()
        camera_d.listen(image_queue_depth.put)
        self.actor_list.append(camera_d)

        #Guardamos informacion del sensor
        image_depth = image_queue_depth.get()
        image_depth.save_to_disk("test_images/%06d_depth.png" %(image_depth.frame), carla.ColorConverter.LogarithmicDepth)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        

#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")

dummy.start()


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    dummy.stop()


quit_spade()
