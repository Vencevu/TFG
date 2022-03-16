from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
import time
import asyncio

from CarlaEnv import CarlaEnv
from DQNEnv import DQNAgent

class CarAgent(agent.Agent):

    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):

        async def on_start(self):
            self.agent_dqn = DQNAgent()
            self.env = CarlaEnv()
            self.env.gen_vehicle()
            await asyncio.sleep(1)

        async def run(self):
            await asyncio.sleep(1)

        async def on_end(self):
            self.env.destroy_all_actors()
            print("Behaviour finished with exit code {}.".format(self.exit_code))

    async def setup(self):
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = CarAgent("agente@localhost", "1234")
dummy.start()

while not dummy.my_behav.is_killed():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        dummy.my_behav.kill()
        time.sleep(2)

dummy.stop()
quit_spade()