from distutils.command.config import config
import warnings
warnings.filterwarnings("ignore")

from spade import agent, quit_spade
from spade.behaviour import CyclicBehaviour
from tqdm import tqdm
import csv
import time
from termcolor import colored
import Config
import numpy as np
import matplotlib.pyplot as plt

from CarlaEnv import CarlaEnv
from DQNEnv import DQNAgent

class CarAgent(agent.Agent):

    #Comportamiento del agente
    class MyBehav(CyclicBehaviour):

        def dqn_car(self):
            epsilon = Config.epsilon
            ep_rewards = [Config.MIN_REWARD]

            #Guardamos los datos en csv
            dist_csv = open('../csv/validate/Distances_%d_%d_%d.csv' % (Config.EPISODES, Config.MINIBATCH_SIZE, Config.REPLAY_MEMORY_SIZE), 'w')
            dist_csv_writer = csv.writer(dist_csv, delimiter=";")
            dist_csv_writer.writerow(['Episodio', 'Distancia', 'Tipo'])

            acc_csv = open('../csv/validate/Acc_%d_%d_%d.csv' % (Config.EPISODES, Config.MINIBATCH_SIZE, Config.REPLAY_MEMORY_SIZE), 'w')
            acc_csv_writer = csv.writer(acc_csv, delimiter=";")
            acc_csv_writer.writerow(['Episodio', 'Precision'])

            loss_csv = open('../csv/validate/Loss_%d_%d_%d.csv' % (Config.EPISODES, Config.MINIBATCH_SIZE, Config.REPLAY_MEMORY_SIZE), 'w')
            loss_csv_writer = csv.writer(loss_csv, delimiter=";")
            loss_csv_writer.writerow(['Episodio', 'Perdida'])

            #Para la grafica
            x_axis_time = []
            x_axis_goal = []
            x_axis_col = []
            accX = [x for x in range(20, Config.EPISODES + 1, 20)]
            y_axis_time = []
            y_axis_col = []
            y_axis_goal = []
            accY = []
            lossY = []

            for self.episode in tqdm(range(1, Config.EPISODES + 1)):
                try:
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
                        if np.random.random() > epsilon:
                            # Get action from Q table
                            action = np.argmax(self.agent_dqn.get_qs(self.current_state))
                        else:
                            # Get random action
                            action = np.random.randint(0, Config.N_ACTIONS)
                            # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                            time.sleep(1 / Config.FPS)

                        distance = round(self.env.distance_to_goal(self.goal_x, self.goal_y), 3)
                        new_state, car_velocity, reward, self.done, reset_type = self.env.step(action, distance)
                        episode_reward += reward
                        self.current_state = new_state
                        self.agent_dqn.update_replay_memory((self.current_state, action, reward, new_state, self.done))
                        
                        step += 1

                        if self.done:
                            reset_type_csv = ''
                            if reset_type == 3:
                                y_axis_time.append(distance)
                                x_axis_time.append(self.episode)
                                reset_type_csv = 'Tiempo'
                            elif reset_type == 2:
                                y_axis_col.append(distance)
                                x_axis_col.append(self.episode)
                                reset_type_csv = 'Colision'
                            elif reset_type == 1:
                                y_axis_goal.append(distance)
                                x_axis_goal.append(self.episode)
                                reset_type_csv = 'Objetivo'
                            
                            dist_csv_writer.writerow([self.episode, distance, reset_type_csv])
                            
                            break
                    
                    ep_rewards.append(episode_reward)
                    if not self.episode % Config.AGGREGATE_STATS_EVERY or self.episode == 1:  ## every show_stats_every, which is 10 right now, show and save teh following
                        average_reward = sum(ep_rewards[-Config.AGGREGATE_STATS_EVERY:]) / len(
                            ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                        min_reward = min(ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                        max_reward = max(ep_rewards[-Config.AGGREGATE_STATS_EVERY:])
                        self.agent_dqn.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                                reward_max=max_reward,
                                                                epsilon=epsilon)

                    if epsilon > Config.MIN_EPSILON:
                        epsilon *= Config.EPSILON_DECAY
                        epsilon = max(Config.MIN_EPSILON, epsilon)

                    if self.episode % 20 == 0:
                        acc, loss = self.agent_dqn.train()
                        accY.append(acc)
                        lossY.append(loss)
                        acc_csv_writer.writerow([self.episode, round(acc[0], 3)])
                        loss_csv_writer.writerow([self.episode, round(loss[0], 3)])
                
                except Exception as e:
                    print("Error en entrenamiento: ",e)
                
            print(colored('End and Save Model...', 'green'))
            self.env.destroy_all_actors()
            self.agent_dqn.save_rl_model()

            plt.scatter(x_axis_time, y_axis_time, label="time reset")
            plt.scatter(x_axis_col, y_axis_col, label="collision reset")
            plt.scatter(x_axis_goal, y_axis_goal, label="goal")
            plt.legend(loc="upper left")
            plt.xlabel("Episodios")
            plt.ylabel("Distancia al objetivo")
            plt.savefig('../graficas/validate/Distances_%d_%d_%d.png' % (Config.EPISODES, Config.MINIBATCH_SIZE, Config.REPLAY_MEMORY_SIZE))
            plt.clf()
            plt.plot(accX, accY)
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.savefig('../graficas/validate/Acc_%d_%d_%d.png' % (Config.EPISODES, Config.MINIBATCH_SIZE, Config.REPLAY_MEMORY_SIZE))
            plt.clf()
            plt.plot(accX, lossY)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig('../graficas/validate/Loss_%d_%d_%d.png' % (Config.EPISODES, Config.MINIBATCH_SIZE, Config.REPLAY_MEMORY_SIZE))
            
            dist_csv.close()
            acc_csv.close()
            loss_csv.close()

            self.episode = 0
            self.kill()            
            
        async def on_start(self):
            self.agent_dqn = DQNAgent()
            self.env = CarlaEnv()

            self.goal_x = Config.GOAL_X
            self.goal_y = Config.GOAL_Y

            self.current_state = self.env.front_camera
            #Cargamos los pesos si hay
            self.agent_dqn.load_rl_model()

        async def run(self):
            self.dqn_car()

        async def on_end(self):
            print("Behaviour finished with exit code {}.".format(self.exit_code))
    
    async def setup(self):
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)


#Lanzamos el agente
dummy = CarAgent("agente@localhost", "1234")
future = dummy.start()
future.result()

while not dummy.my_behav.is_killed():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        dummy.stop()
        break

print("Agent %s finished" % dummy.name)

quit_spade()