""" Trains an agent with (stochastic) Policy Gradients on Mountain Car. Uses OpenAI Gym. """
import gym
from agent import Agent
import numpy as np
import getopt, sys


game = "MountainCar-v0"
env = gym.make(game)

# hyperparameters
I = 2 # inputs
O = 3 # outputs
H = 4 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True   # resume from previous checkpoint?
render = False
update_method = "adagrad"
biased = False


possible_update_methods = ["adagrad", "rmscache"]

opts, args = getopt.getopt(sys.argv[1:], "u:b:l:r:h:g:a:")
for opt, arg in opts:
	if opt == "-u":
		if arg in possible_update_methods:
			update_method = arg
		else:
			print "Not a correct update method, uses adagrad instead."

	elif opt == "-b":
		batch_size = arg
	elif opt == "-l":
		learning_rate = float(learning_rate)
	elif opt == "-r":
		resume = arg == "1"
	elif opt == "-h":
		H = int(arg)
	elif opt == "-g":
		gamma = float(arg)
	elif opt == "-a":
		biased = arg == "1"





agent = Agent(I, H, O, batch_size, learning_rate, gamma, decay_rate, "{}-H{}-U_{}{}.p".format(game, H,update_method, ("-B_1" if biased else "")), update_method, game, biased)

if resume:
  agent.load_model()
else:
  agent.new_model()


agent.init_gradbuffer()
agent.init_rmscache()
agent.init_adagrad_mem()


observation = env.reset()


tick = 0
while True:
  tick += 1
  if render: env.render()

  # step the environment and get new measurements
  observation, reward, done, info = env.step( agent.decide_action(observation))
  agent.register_reward(reward)

  if( tick >= 500000):
    agent.episode_restarted()
    tick = 0
    observation = env.reset()
    reward_sum = 0


  if done: # an episode finished
    tick = 0
    agent.episode_finished()
    observation = env.reset() # reset env

