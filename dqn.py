
# coding: utf-8



import sys
import gym
import keras
import random
import math
import numpy as np
from collections import deque



n_episodes=1000
n_win_ticks=195
max_env_steps=None

gamma= 1.0
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
alpha = 0.02 #learning_rate
alpha_decay = 0.01


batch_size = 64
monitor= False
quiet = False

#envt parameters
memory = deque(maxlen=100000)
env = gym.make('CartPole-v0')
if max_env_steps is not None: env.max_episode_steps = max_env_steps


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#Model Definition
model = Sequential()
model.add(Dense(24,input_dim=4,activation = 'relu'))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(2,activation = 'relu'))
model.compile(loss='mse',optimizer=Adam(lr=alpha,decay = alpha_decay))
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


def remember(state,action,reward,next_state,done):
	memory.append((state,action,reward,next_state,done))


def choose_action(state,epsilon):
	if np.random.random() <= epsilon:
		return env.action_space.sample()
	else: 
		#print("Model predicted:",model.predict(state))
		return np.argmax(model.predict(state))



def get_epsilon(t):
	#print()
	ep=max(epsilon_min,min(epsilon,1.0 - math.log10((t+1)* epsilon_decay)))
	#print("epsilon:",ep)
	return ep
def preprocess_state(state):
	return np.reshape(state,[1,4])

def replay(batch_size,epsilon):
	x_batch=[]
	y_batch=[]
	minibatch = random.sample(memory,min(len(memory),batch_size))
	
	for state,action,reward,next_state,done in minibatch:
		y_target = model.predict(state)
		y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
		x_batch.append(state[0])
		y_batch.append(y_target[0])
		
	model.fit(np.array(x_batch),np.array(y_batch),batch_size=len(x_batch),verbose=0)
	
	if epsilon>epsilon_min:
		epsilon*=epsilon_decay
		
		
		
def run():
	scores=deque(maxlen=100)
	
	for e in range(n_episodes):
		state = preprocess_state(env.reset())
		done= False
		i=0
		while not done:
			#print("state:",state)
			action=choose_action(state,get_epsilon(e))
			next_state,reward,done, _ = env.step(action)
			env.render()
			next_state= preprocess_state(next_state)
			remember(state,action,reward,next_state,done)
			state=next_state
			i+=1
			
		scores.append(i)
		#print(scores)
		mean_score=np.mean(scores)
		
		
		if mean_score >= n_win_ticks and e>=100:
			if not quiet: print('Ran:{} episodes.Solved after {} trails'.format(e,e-100))
			return e-100
		
		if e%20==0 and not quiet :
			#print(get_epsilon(e))
			print('[Epsiode{}] - Mean survival time over last 100 episodes as {} ticks.'.format(e,mean_score))
			
		
		replay(batch_size,get_epsilon(e))
		
	if not quiet: print('Did not solveafter {} episodes'.format(e))
	return e
			
		
		
run()




	










