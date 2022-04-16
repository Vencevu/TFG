import carla
import asyncio
import random
import time
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
            bp = random.choice(blueprint_library.filter('vehicle'))
            transform = self.map.get_spawn_points()[0]
            transform.location.x = -35
            
            self.vehicle = self.world.spawn_actor(bp, transform) 
            self.actor_list.append(self.vehicle)
            
            await asyncio.sleep(1)

            #Gestor de conduccion
            args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
            args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}

            
            self.PID = VehiclePIDController(self.vehicle,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
            
            vehicle_wp = self.map.get_waypoint(self.vehicle.get_location())
            self.next_wp = vehicle_wp.next(10)[0]
            control = self.PID.run_step(5, self.next_wp)
            self.vehicle.apply_control(control)
            

        async def run(self):
            control = self.PID.run_step(5, self.next_wp)
            self.vehicle.apply_control(control)

            vehicle_loc = self.vehicle.get_location()
            print("Posicion y ", vehicle_loc.y)
            vehicle_wp = self.map.get_waypoint(vehicle_loc)
            dist = self.next_wp.transform.location.distance(vehicle_loc)
            print("Distancia: ", dist)
            if dist < 1:
                print("Cerca del siguiente punto")
                posibles_puntos = vehicle_wp.next(10)
                print("Posibles puntos:")
                print("--------------------------------------------------")
                for p in posibles_puntos:
                    print(p.transform.location.__str__())
                print("--------------------------------------------------")
                self.next_wp = posibles_puntos[0]
               
            await asyncio.sleep(0.5)

        async def on_end(self):
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            print("Behaviour finished with exit code {}.".format(self.exit_code))

    my_behav = MyBehav()
    async def setup(self):
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
