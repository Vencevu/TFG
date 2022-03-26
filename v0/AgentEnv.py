from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
from tqdm import tqdm
import time
from termcolor import colored
import Config
import numpy as np
import asyncio

from CarlaEnv import CarlaEnv
from DQNEnv import DQNAgent

class CarAgent(agent.Agent):

    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):

        def dqn_car(self):
            epsilon = 0
            ep_rewards = [Config.MIN_REWARD]

            for self.episode in tqdm(range(1, Config.EPISODES + 1)):
                self.env.reset()
                episode_reward = 0
                step = 1


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
                        action = np.random.randint(0, 4)
                        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                        time.sleep(1 / Config.FPS)
                    
                    distance = self.env.distance_to_goal(self.goal_x, self.goal_y)
                    new_state, car_velocity, reward, self.done, _ = self.env.step(action, distance)
                    episode_reward += reward
                    self.current_state = new_state
                    self.agent_dqn.update_replay_memory((self.current_state, action, reward, new_state, self.done))

                    step += 1

                    if self.done:
                        break
                
                ep_rewards.append(episode_reward)
                if not self.episode % Config.AGGREGATE_STATS_EVERY or self.episode == 1:  ## every show_stats_every, which is 10 right now, show and save teh following
                    average_reward = sum(ep_rewards[-Config.AGGREGATE_STATS_EVERY:]) / len(
                        ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                    self.agent_dqn.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                            reward_max=max_reward,
                                                            epsilon=Config.epsilon)

                if epsilon > Config.MIN_EPSILON:
                    epsilon *= Config.EPSILON_DECAY
                    epsilon = max(Config.MIN_EPSILON, epsilon)

            print(colored('End and Save Model...', 'green'))

            self.agent_dqn.save_rl_model()
            
            self.env.destroy_all_actors()
            self.episode = 0
            self.kill()

        async def on_start(self):
            self.agent_dqn = DQNAgent()
            self.env = CarlaEnv()

            self.goal_x = -36
            self.goal_y = 24

            self.current_state = self.env.front_camera

            

        async def run(self):
            self.dqn_car()

        async def on_end(self):
            self.env.destroy_all_actors()
            print("Behaviour finished with exit code {}.".format(self.exit_code))
    
    async def setup(self):
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = CarAgent("agente@localhost", "1234")
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