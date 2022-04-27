from pydoc import describe
from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
import asyncio
import carla
import random
import time

#Clase Agente
class DummyAgent(agent.Agent):
    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):

        actor_list = []
        world = None
        client = None

        async def on_start(self):
            #Conexion con CARLA
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(10.0)

            #Configuracion del mundo
            self.world = self.client.get_world()

            #Creamos vehiculo
            blueprint_library = self.world.get_blueprint_library()
            bp = blueprint_library.filter('vehicle')[0]
            transform = self.world.get_map().get_spawn_points()[0]
            vehicle = self.world.spawn_actor(bp, transform) 
            self.actor_list.append(vehicle)

            #Creamos sensor y lo acoplamos al vehiculo
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('upper_fov', str(15))
            lidar_bp.set_attribute('lower_fov', str(-25))
            lidar_bp.set_attribute('channels', str(64))
            lidar_bp.set_attribute('range', str(30))
            lidar_bp.set_attribute('rotation_frequency', str(20))
            lidar_bp.set_attribute('points_per_second', str(500000))

            lidar_location = carla.Location(0, 0, 1.5)
            lidar_transform = carla.Transform(lidar_location)
            lidar_sen = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle)

            lidar_sen.listen(self.save_lidar)

            self.actor_list.append(lidar_sen)

        async def run(self):
            await asyncio.sleep(5)
            self.kill()

        async def on_end(self):
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            print("Behaviour finished with exit code {}.".format(self.exit_code))

        def save_lidar(self, data):
            data.save_to_disk('test_images/lidar/%.6d.ply' % data.frame)

    async def setup(self):
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")
future = dummy.start()
time.sleep(1)
future.result()

while dummy.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        dummy.stop()
        break

print("Agents finished")

quit_spade()
