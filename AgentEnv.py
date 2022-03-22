from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
from tqdm import tqdm
import time
import Config
import numpy as np
import asyncio

from CarlaEnv import CarlaEnv
from DQNEnv import DQNAgent

class CarAgent(agent.Agent):

    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):

        def dqn_car(self):
            for self.episode in tqdm(range(1, Config.EPISODES + 1)):
                # try:
                # Restarting episode - reset episode reward and step number
                self.episode_reward = 0
                self.step = 1

                # Update tensorboard step every episode
                self.agent_dqn.tensorboard.step = self.episode

                # Reset flag and start iterating until episode ends
                self.done = False

                episode_start = time.time()
                # Play for given number of seconds only
                while True:

                    # np.random.random() will give us the random number between 0 and 1. If this number is greater than
                    # our randomness variable,
                    # we will get Q values based on tranning, but otherwise, we will go random actions.
                    if np.random.random() > Config.epsilon:
                        # Get action from Q table
                        action = np.argmax(self.agent_dqn.get_qs(self.current_state))
                    else:
                        # Get random action
                        action = np.random.randint(0, 3)
                        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                        time.sleep(1 / Config.FPS)
                    
                    new_state, car_velocity, reward, self.done, _ = self.env.step(action)

                    self.current_state = new_state
                    self.agent_dqn.update_replay_memory((self.current_state, action, reward, new_state, self.done))

                    self.step += 1

                    if self.done:
                        break
                
                self.ep_rewards.append(self.episode_reward)
                if not self.episode % Config.AGGREGATE_STATS_EVERY or self.episode == 1:  ## every show_stats_every, which is 10 right now, show and save teh following
                    average_reward = sum(self.ep_rewards[-Config.AGGREGATE_STATS_EVERY:]) / len(
                        self.ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                    min_reward = min(self.ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                    max_reward = max(self.ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                    self.agent_dqn.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                            reward_max=max_reward,
                                                            epsilon=Config.epsilon)
            
            self.agent_dqn.save_rl_model()

            self.episode = 0
            self.end_dqn = False

        async def on_start(self):
            self.agent_dqn = DQNAgent()
            self.env = CarlaEnv()

            self.env.gen_vehicle()
            await asyncio.sleep(1)
            self.env.add_sensor("rgb_cam")
            await asyncio.sleep(1)
            
            self.epsilon = 0
            self.ep_rewards = [Config.MIN_REWARD]
            self.current_state = self.env.front_camera

            

        async def run(self):
            self.dqn_car()

        async def on_end(self):
            self.env.destroy_all_actors()
            print("Behaviour finished with exit code {}.".format(self.exit_code))

    my_behav = MyBehav()
    
    async def setup(self):
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