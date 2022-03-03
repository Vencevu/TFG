import carla
import asyncio
import random
import time
from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
from CARLA.PythonAPI.carla.agents.navigation.controller import VehiclePIDController

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
            map = self.world.get_map()

            #Creamos vehiculo
            blueprint_library = self.world.get_blueprint_library()
            bp = random.choice(blueprint_library.filter('vehicle'))
            transform = map.get_spawn_points()[0]
            vehicle = self.world.spawn_actor(bp, transform) 
            self.actor_list.append(vehicle)

            wp = map.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            args_lateral_dict = {
                'K_P': 1.95,
                'K_D': 0.2,
                'K_I': 0.07,
                'dt': 1.0 / 10.0
            }

            args_long_dict = {
                'K_P': 1,
                'K_D': 0.0,
                'K_I': 0.75,
                'dt': 1.0 / 10.0
            }

            PID = VehiclePIDController(vehicle,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
            control = PID.run_step(10, wp)
            vehicle.apply_control(control)

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
