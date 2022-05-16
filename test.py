import carla
import asyncio
import random
import time
from cv2 import transform
from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
from agents.navigation.controller import VehiclePIDController

#Clase Agente
class DummyAgent(agent.Agent):

    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):

        actor_list = []
        next_wp = None
        world = None
        map = None
        client = None
        vehicle = None
        PID = None

        async def on_start(self):
            #Conexion con CARLA
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(10.0)

            #Configuracion del mundo
            self.world = self.client.get_world()
            self.map = self.world.get_map()

            #Creamos vehiculo
            blueprint_library = self.world.get_blueprint_library()
            bp = blueprint_library.filter('vehicle')[0]
            transform = carla.Transform(carla.Location(-30, 25, 0.6))

            self.vehicle = self.world.spawn_actor(bp, transform) 
            self.actor_list.append(self.vehicle)
            
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(320))
            bp.set_attribute('image_size_y', str(240))
            bp.set_attribute('fov', '110')
            bp.set_attribute('sensor_tick', '1.0')
            transform = carla.Transform(carla.Location(x=0.8, z=1.5))
            sensor = self.world.try_spawn_actor(bp, transform, attach_to=self.vehicle)

            self.actor_list.append(sensor)
            sensor.listen(lambda data: self.process_img(data))
            time.sleep(1)
            self.kill()
        
        def process_img(self, image):
            image.save_to_disk('test_images/rgb_cam/%d.png' % image.frame)

        async def run(self):
            await asyncio.sleep(1)

        async def on_end(self):
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            print("Behaviour finished with exit code {}.".format(self.exit_code))

    my_behav = MyBehav()
    async def setup(self):
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")
future = dummy.start()
future.result()

while not dummy.my_behav.is_killed():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        dummy.top()
        break

quit_spade()
