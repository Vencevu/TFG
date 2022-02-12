from spade import agent, quit_spade
import carla

#Clase Agente
class DummyAgent(agent.Agent):
    async def setup(self):
        #Conexion con CARLA
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)

        world = client.get_world()

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print(number_of_spawn_points)

        print("Hello World! I'm agent {}".format(str(self.jid)))



#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")
future = dummy.start()
future.result()

dummy.stop()
quit_spade()
