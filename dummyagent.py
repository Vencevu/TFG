from spade import agent, quit_spade

class DummyAgent(agent.Agent):
    async def setup(self):
        print("Hello World! I'm agent {}".format(str(self.jid)))

dummy = DummyAgent("agente@localhost", "1234")
future = dummy.start()
future.result()

dummy.stop()
quit_spade()