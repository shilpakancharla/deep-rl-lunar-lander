import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

class DeepQLearning:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.epsilon_decay = 0.996
        self.memory = deque(maxlen = 1000000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim = self.state_space, activation = relu))
        model.add(Dense(120, activation = relu))
        model.add(Dense(self.action_space, activation = linear))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

    def remember(self, state, action, reward, next_state, done_status):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Exploration-exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.array(i[0] for i in mini_batch)
        actions = np.array(i[1] for i in mini_batch)
        rewards = np.array(i[2] for i in mini_batch)
        next_states = np.array(i[3] for i in mini_batch)
        done_status = np.array(i[4] for i in mini_batch)

        states = np.squeeze(states) # Remove axes of length one from states
        next_states = np.squeeze(next_states) # Remove axes of length one from next_states

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis = 1)) * (1 - done_status)
        targets_full = self.model.predict_on_batch(states)
        idx = np.array([i for i in range(self.batch_size)])
        targets_full[[idx], [actions]] = targets

        self.model.fit(states, targets_full, epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(episode):
    loss = []
    agent = DeepQLearning(env.action_space.n, env.observation_space.shape[0])
    for ep in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done_status, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done_status)
            state = next_state
            agent.replay()
            if done_status: # Status is true
                print("Episode: {}/{}, Score: {}".format(ep, episode, score))
                break
        loss.append(score)
        is_solved = np.mean(loss[-100:]) # Average score of last 100 episodes
        if is_solved > 200:
            print('\nTask Completed.\n')
            break
        print("Average over last 100 episodes: {0:.2f}\n".format(is_solved))
    return loss

if __name__ == '__main__':
    print(env.observation_space)
    print(env.action_space)
    episodes = 400
    loss = train(episodes)
    plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()