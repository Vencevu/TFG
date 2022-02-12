from spade import agent, quit_spade
import carla

#Clase Agente
class DummyAgent(agent.Agent):
    async def setup(self):
        print("Hello World! I'm agent {}".format(str(self.jid)))

#Conexion con CARLA
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

world = client.get_world()

#Lanzamos el agente
dummy = DummyAgent("agente@localhost", "1234")
future = dummy.start()
future.result()

dummy.stop()
quit_spade()
