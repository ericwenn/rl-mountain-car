""" Trains an agent with (stochastic) Policy Gradients on Mountain Car. Uses OpenAI Gym. """
import gym
from recurrentagent import Agent
import numpy as np
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


agent = Agent(I, H, O, batch_size, learning_rate, gamma, decay_rate, "{}-H{}-recurrent.p".format(game, H))

if resume:
  agent.load_model()
else:
  agent.new_model()


agent.init_gradbuffer()
agent.init_rmscache()


observation = env.reset()


running_reward = None
reward_sum = 0
tick = 0

last_observation = np.zeros_like(observation)
while True:
  tick += 1
  if render: env.render()

  # step the environment and get new measurements
  observation, reward, done, info = env.step( agent.decide_action(observation))
  reward_sum += reward

  agent.register_reward(reward)

  if( tick >= 25000):
    agent.episode_restarted()
    tick = 0
    observation = env.reset()
    reward_sum = 0


  if done: # an episode finished
    tick = 0
    agent.episode_finished()

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)

    reward_sum = 0
    observation = env.reset() # reset env

