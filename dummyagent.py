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
            bp = random.choice(blueprint_library.filter('vehicle'))
            transform = self.world.get_map().get_spawn_points()[0]
            vehicle = self.world.spawn_actor(bp, transform) 
            self.actor_list.append(vehicle)

            vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.2))


        async def run(self):
            await asyncio.sleep(1)

        async def on_end(self):
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            print("Behaviour finished with exit code {}.".format(self.exit_code))

    my_behav = MyBehav()
    async def setup(self):
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")
dummy.start()

while not dummy.my_behav.is_killed():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        dummy.my_behav.kill()
        time.sleep(2)

dummy.stop()
quit_spade()
