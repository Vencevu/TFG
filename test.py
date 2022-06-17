from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
import time

class CarAgent(agent.Agent):

    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):
        current_state = None
    
        async def on_start(self):
            print("Comportamiento iniciado")

        async def run(self):
            print("Ejecucion")

        async def on_end(self):
            print("Behaviour finished with exit code {}.".format(self.exit_code))
    
    async def setup(self):
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = CarAgent("agente@localhost", "1234")
future = dummy.start()
future.result()

while dummy.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        dummy.stop()
        break

print("Agent %s finished" % dummy.name)

quit_spade()