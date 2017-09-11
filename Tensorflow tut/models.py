import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Actor(Model):
    def __init__(self,action_dims,name='actor'):
        super(Actor, self).__init__(name=name)
        self.action_dims = action_dims

    def __call__(self, input_observation, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = input_observation
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.action_dims, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x

class Critic(Model):
    def __init__(self,observation_dims,action_dims,name='critic'):
        super(Critic, self).__init__(name=name)
        self.observation_dims = observation_dims
        self.action_dims = action_dims
    def __call__(self, input_observation, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = input_observation
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x 

def get_target_updates(vars, target_vars, tau):
    #logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        #logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)