import gym
import numpy

numpy.bool8 = numpy.bool_

#to create the environment for the agent to interact.

env = gym.make("LunarLander-v2", render_mode = 'human')   #render mode human allows you to watch agent interacting on the screen.

#to get the possible observations and actions.
print(f"states:{env.observation_space}")
print(f"actions:{env.action_space}")

#to start the simulation.
env.reset()

for i in range(1000):

     #all possible actions in the particular observations(state)
     actions = env.action_space.sample()

     #taking the action from the above possible actions
     policy = env.step(actions)

     #getting the reward next state adn actions from the past taken actions

     state, reward , done , info ,addi_info = policy

     # after rollout of episode
     if done:
          env.reset()
          print(f"state:{state}")
          print(f"actions{reward}")

env.close()
