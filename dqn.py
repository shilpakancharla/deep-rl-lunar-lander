import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

"""
    Implementation of Deep Q-Learning Network (DQN).
"""
class DeepQLearning:
    """
        @action_space: Four discrete actions are available: do nothing, fire left orientation engine, 
                       fire main engine, fire right orientation engine.
        @state_space: State refers to position of the lunar lander. Landing site has coordinates (0, 0).
        @batch_size: Determines the batch size.
        @epsilon_min: Minimum allowed epsilon value. Determines if epsilon decay is factored in.
        @epsilon: Randomness factor which forces us to try different actions.
        @gamma: Discount factor which quantifies how much importance we give for future rewards.
        @learning_rate: Determines the step size at each iteration while moving toward a minimum of a loss function.
        @epsilon_decay: Factor by which epsilon decreases.
        @record: List to keep track of state, reward, action, the next state, and if we have reach a terminal state.
        @model: Calls to define the agent model for DQN.

        We conduct a search of the best hyperparameters based on the epsilon, gamma, learning rate, and epsilon decay.
        We initialize a model that takes into account these hyperparameters.
    """
    def __init__(self, action_space, state_space, epsilon, gamma, learning_rate, epsilon_decay):
        self.action_space = action_space
        self.state_space = state_space
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.record = list() # Create a new record for every combination of parameters
        self.model = self.define_agent_model()

    """
        Defining the agent's neural network.

        @return compiled neural network model
    """
    def define_agent_model(self):
        model = Sequential()
        model.add(Dense(125, input_dim = self.state_space, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(self.action_space, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model

    """
        Keeping a record of the state, action, reward, next state, and if we have reached the terminal
        status.

        @param state: current state
        @param action: action taken
        @param reward: reward earned from action taken
        @param next state: next state in the state space from action taken
        @param terminal_status: Boolean flag to see if we have reached the terminal state
    """
    def keep_record(self, state, action, reward, next_state, terminal_status):
        self.record.append((state, action, reward, next_state, terminal_status))
    
    """
        Choose an action at random and then predict based on the state from the model we have created after training.
        
        @param state: current state
        @return action that yields the greatest reward
    """
    def act(self, state):
        # Exploration-exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.record) < self.batch_size:
            return
        
        mini_batch = random.sample(self.record, self.batch_size)
        states = np.array([i[0] for i in mini_batch])
        actions = np.array([i[1] for i in mini_batch])
        rewards = np.array([i[2] for i in mini_batch])
        next_states = np.array([i[3] for i in mini_batch])
        terminal_status = np.array([i[4] for i in mini_batch])

        states = np.squeeze(states) # Remove axes of length one from states
        next_states = np.squeeze(next_states) # Remove axes of length one from next_states

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis = 1)) * (1 - terminal_status)
        targets_full = self.model.predict_on_batch(states)
        idx = np.array([i for i in range(self.batch_size)])
        targets_full[[idx], [actions]] = targets

        self.model.fit(states, targets_full, epochs = 1, verbose = 0)
        # If the current epsilon value is greater than the minimum allowed epsilon value, decay the value
        # If the current epsilon value is less than the minimum already, do not decay it further
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

"""
    We can modify the possible hyperparameters we need to search from here to train the network and calculate
    the rewards and loss values.

    @param episodes: number of episodes we want to conduct
    @param ev: epsilon value (randomness factor)
    @param gv: discount factor value
    @param lrv: learning rate value
    @param edv: epsilon-decay value
    @return loss values of the function
"""
def train_dqn(episode, ev, gv, lrv, edv):
    loss = [] # Create new array that will hold the loss value
    agent = DeepQLearning(env.action_space.n, env.observation_space.shape[0], ev, gv, lrv, edv) # Define the agent parameters
    for ep in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 2000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, terminal_status, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.keep_record(state, action, reward, next_state, terminal_status)
            state = next_state
            agent.replay()
            if terminal_status: # If we have reach the terminal state
                f = open("log_files\log_dqn_{}_{}_{}_{}.txt".format(ev, gv, lrv, edv), "a")
                f.write("\nEpisode: {}/{}, Score: {}".format(ep, episode, score))
                f.close()
                break
        loss.append(score)
        last_hundred_avg = np.mean(loss[-100:]) # Average score of last 100 episodes
        # This average would be a good indicator of a decent set of hyperparameters
        if last_hundred_avg > 200:
            f = open("log_files\log_dqn_{}_{}_{}_{}.txt".format(ev, gv, lrv, edv), "a")
            f.write('\nTask completed.\n')
            f.close()
            break
        # Break if the average becomes very low - not a good set of hyperparameters
        elif last_hundred_avg < -800: 
            break
        f = open("log_files\log_dqn_{}_{}_{}_{}.txt".format(ev, gv, lrv, edv), "a")
        f.write("\nAverage of last 100 episodes: {0:.2f}\n".format(last_hundred_avg))
        f.close()
    return loss
    
if __name__ == '__main__':
    # Setting and seeding environmental parameters
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    # Possible hyperparameter values to search from
    epsilon_values = [1.0, 0.5, 0.3, 0.1]
    gamma_values = [1.0, 0.5, 0.3, 0.1]
    learning_rate_values = [0.0001]
    epsilon_decay_values = [0.995, 0.900]

    for ev in epsilon_values:
        for gv in gamma_values:
            for lrv in learning_rate_values:
                for edv in epsilon_decay_values:
                    loss = train_dqn(400, ev, gv, lrv, edv) # Train for 400 episodes

                    # Create a plot for the loss in the trial
                    plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
                    plt.title("Loss: epsilon = {}, gamma = {}, alpha = {}, epsilon-decay = {}".format(ev, gv, lrv, edv))
                    plt.xlabel("Steps")
                    plt.ylabel("Loss")
                    plt.savefig("figures/loss/loss_{}_{}_{}_{}.png".format(ev, gv, lrv, edv))
                    plt.clf() # Clear figure