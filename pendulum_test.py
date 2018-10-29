import numpy as np
import math as m
import gym
from gym import spaces
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# env = gym.make('Pendulum-v0')
# training_data = np.empty(9)

# for i_episode in range(5):
#     observation = env.reset()
#     theta = m.acos(observation[0])
#     s0 = np.array([observation[0], observation[1], theta, observation[2]])
#     for t in range(420):
#         env.render()  
#         action = [0]
#         sa0 = np.hstack((s0,action))
#         observation, reward, done, info = env.step(action)

#         theta = m.acos(observation[0])
#         s1 = np.array([observation[0], observation[1], theta, observation[2]])
#         training_data = np.vstack((training_data,np.hstack((sa0,s1))))
#         s0 = s1
#         if done:
#             # print("Episode finished after {} timesteps".format(t+1))
#             break
            
# env.close()

sa0 = np.array([1,2,3,4,5])
# sa0.reshape(-1,len(sa0))
print(sa0)