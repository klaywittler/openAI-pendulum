import numpy as np
import math as m
import gym
from gym import spaces

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

env = gym.make('Pendulum-v0')

# initialization variables
LR = 1e-3 # learning rate
goal_steps = 500 # how long each game runs
initial_games = 500 # how many games
test_games = 5 # how test games
test_steps = 500 # how long each test game runs

def get_trainingData(episodes=20,goal_steps=100):
    training_data = np.empty(9)

    for i_episode in range(episodes):
        observation = env.reset()
        # state vector:[ cos(theta)   ,  sin(theta)   ,  theta                , theta dot     ]
        s0 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
        for t in range(goal_steps):
            # env.render()
            action = env.action_space.sample()
            sa0 = np.hstack((s0,action))
            observation, reward, done, info = env.step(action)
            
            s1 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
            training_data = np.vstack((training_data,np.hstack((sa0,s1))))
            s0 = s1
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
                
    env.close()
    training_data = np.delete(training_data,0,axis=0)
    return training_data


def get_trainingPrediction(training_data,model):
    training_predict = np.empty(4)

    for row in training_data[:,0:5]:
        training_predict = np.vstack((training_predict,model.predict(row.reshape(-1,len(row)))))

    training_predict = np.delete(training_predict,0,axis=0)
    return training_predict


def get_testingData(model,test_games=5,test_steps=100):
    testing_predict = np.empty(4)
    testing_actual = np.empty(4)
    for i_episode in range(test_games):
        observation = env.reset()
        # state vector:[ cos(theta)   ,  sin(theta)   ,  theta                , theta dot     ]
        s0 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
        for t in range(goal_steps):
            # env.render()
            action = env.action_space.sample()
            sa0 = np.hstack((s0,action))
            predict_state = model.predict(sa0.reshape(-1,len(sa0)))
            observation, reward, done, info = env.step(action)
            
            s1 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
            testing_predict = np.vstack((testing_actual,s1))
            testing_predict = np.vstack((testing_predict,predict_state))
            s0 = s1
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    testing_predict = np.delete(testing_predict,0,axis=0)
    return [testing_predict, testing_actual]


def neural_network_model(input_size):
    drpout = 0.7
    network = input_data(shape=[None,input_size], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, drpout)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, drpout)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, drpout)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, drpout)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, drpout)

    network = fully_connected(network, 4, activation='relu')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='mean_square', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=3)
    return model


def train_model(training_data, model=False):
    X = training_data[:,0:5]
    y = training_data[:,5:9]

    if not model:
        model = neural_network_model(input_size = len(X[0,:]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning1')
    return model


def get_costMap(model, n_theta = 25, n_dtheta = 25, n_u = 25):
    theta_lower = -m.pi
    theta_upper = m.pi
    dtheta_lower = -8
    dtheta_upper = 8
    u_lower = -2
    u_upper = 2

    theta_space = np.linspace(theta_lower,theta_upper,n_theta)
    dtheta_space = np.linspace(dtheta_lower,dtheta_upper,n_dtheta)
    action_space = np.linspace(u_lower,u_upper,n_u)

    cost = np.empty([theta_space.size,dtheta_space.size,action_space.size])

    t=0
    for theta in theta_space:
        dt = 0
        for dtheta in dtheta_space:
            a = 0
            for action in action_space:
                sa0 = np.array([m.cos(theta), m.sin(theta), theta, dtheta, action])
                predict_state = model.predict(sa0.reshape(-1,len(sa0)))
                # heuristic
                h = -(predict_state[0][2]**2 + predict_state[0][3]**2 + 0.001*action**2)
                cost[t,dt,a] = h
                a += 1
            dt += 1
        t += 1  

    return [cost, theta_space, dtheta_space, action_space]


def simulation(model):
    [cost, theta_space, dtheta_space, action_space] = get_costMap(model)
    for i_episode in range(5):
        observation = env.reset()
        for t in range(10000):
            env.render()
            theta = m.acos(observation[0]) 
            dtheta = observation[2]
            t = np.where(theta_space>=theta) 
            dt = np.where(dtheta_space>=dtheta)
            a = cost[t[0][0],dt[0][0],:].argmax()
            action = [action_space[a]]
            observation, reward, done, info = env.step(action)
            
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
                    
    env.close()


# uncomment this and comment below to get new training data
# training_data = get_trainingData(initial_games,goal_steps)
# np.save('saved_training_data0.npy',training_data)

# loading previously saved data
training_data = np.load('saved_training_data.npy')
X = training_data[:,0:5]
y = training_data[:,5:9]

# uncomment this to retrain model
model = train_model(training_data)
model.save('pendulum1.tflearn')

# loading previously saved model
# model = neural_network_model(len(X[0,:]))
# model.load('pendulum.tflearn')

# uncomment this and comment below to recalculate
# training_predict = get_trainingPrediction(training_data,model)
# np.save('saved_training_prediction.npy',training_predict)
# testing_data = get_testingData(model,test_games,test_steps)
# np.save('saved_testing_data.npy',testing_data)

# loading previously saved data
# training_predict = np.load('saved_training_prediction.npy')
# testing_data = np.load('saved_testing_data.npy')

# getting error
# training_error = y - training_predict
# testing_error = testing_data[1] - testing_data[0]

simulation(model)


