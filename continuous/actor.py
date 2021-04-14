import tflearn
import tensorflow as tf
from tflearn.initializations import uniform

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_parameters = tf.trainable_variables() 

        # Target network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_parameters = tf.trainable_variables()[len(self.network_parameters):]

        # Periodically update target network with online network
        self.update_target_network_parameters =\ 
            [self.target_network_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau) +
                                                    tf.multiply(self.target_network_parameters[i], 1. - self.tau))
            for i in range(self.target_network_parameters)]

        # Gradient will be provided by Critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        # Combine gradients
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_parameters, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape = [None, self.state_dim])

        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        net = tflearn.fully_connected(net, 200, weights_init = uniform(minval = 0.002, maxval = 0.002))
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        out = tflearn.fully_connected(net, self.action_dim, activation = 'tanh')

        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, target_scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict = {
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict = {
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict = {
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
        