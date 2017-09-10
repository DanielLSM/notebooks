'''A complete DDPG agent, everything running on tensorflow should just run 
in this class for sanity and simplicity. Moreoever, every variable and
hyperparameter should be stored within the tensorflow graph to grant
increased performance'''
#agent=DDPG_agent(something,something)....
from canton.misc import get_session
import tensorflow as tf
from copy import copy

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


class DDPG_agent(object):
    
    def __init__(self,observation_dims=1, action_dims=1,
        alpha=0.9,gamma=0.99, noise=0.2,memory_size=1e6,batch_size=64,tau=1.,
        actor_alpha=1e-4,critic_alpha=1e-4):
        
        #Inputs
        self.observation = tf.placeholder(tf.float32, shape=(None,observation_dims), name='observation')
        self.action = tf.placeholder(tf.float32, shape=(None,action_dims), name='action')        
        self.observation_after = tf.placeholder(tf.float32, shape=(None,observation_dims), name='observation_after')
        self.reward = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')        
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        
        #self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        #self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
 
        #Hyper Parameters
        self.alpha = alpha
        self.noise = noise
        self.gamma = gamma
        self.tau = tau
        self.memory = memory_size
        self.batch_size = batch_size


        #Networks
        self.actor = Actor(action_dims)
        self.target_actor = Actor(action_dims,name='target_actor')
        self.critic = Critic(observation_dims,action_dims)
        self.target_critic = Critic(observation_dims,action_dims,name='target_critic')


        #Expose nodes from the tf graph to be used

        # Critic Nodes
        self.a2 = self.target_actor(self.observation_after)
        self.q2 = self.target_critic(self.observation_after , self.a2)
        self.q1_target = self.reward + (1-self.terminals1) * self.gamma * self.q2
        self.q1_predict = self.critic(self.observation,self.action)
        critic_loss = tf.reduce_mean((self.q1_target - self.q1_predict)**2)

        # Actor Nodes
        self.a1_predict = self.actor(self.observation)
        self.q1_predict = self.critic(self.observation,self.a1_predict,reuse=True)
        actor_loss = tf.reduce_mean(- self.q1_predict) 

        # Infer
        self.a_infer = self.actor(self.observation,reuse=True)
        self.q_infer = self.critic(self.observation,self.a_infer,reuse=True)


        # Sync Networks
        self.initialize()
        self.setup_target_network_updates()
        self.sync_target()




        #Train Boosters
        #self.replay_memory =
        #self.actor, self.actor_target = Actor()
        #self.critic, self.critic_target = Critic()

    def __call__(self,input_observation):
        feed_dict = {self.observation: [input_observation]}
        #actor = self.actor
        #obs = np.reshape(observation,(1,len(observation)))###############
        sess = get_session()
        action = sess.run(self.a_infer,feed_dict=feed_dict)
        #action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        #action = action.flatten()
        #if self.action_noise is not None and apply_noise:
        #    noise = self.action_noise()
        #    assert noise.shape == action.shape
        #    action += noise
        #action = np.clip(action, self.action_range[0], self.action_range[1])
        return action #action, q

        

    def __len__(self):
        #return memory_replay_size and/or number of episodes
        pass 
    def initialize(self):
        sess = get_session()
        sess.run(tf.global_variables_initializer())
        #self.actor_optimizer.sync()
        #self.critic_optimizer.sync()
        #self.sess.run(self.target_init_updates)

    #@property
    def get_global_variables(self,name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)

    def get_trainable_vars(self,name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def update_targets(self):
        pass
    
    def prepare_train_graph_ops(self):
        pass
        #return something1,something2
    def feed_sample(self):
        #self.replay_memory.
        pass 

    def train(self):
        #batch something self.replay_memory.batch
        #feed some dict to a node on the graph
        pass
        
    def load_agent(self):
        loader = tf.train.Saver()
        sess = get_session()
        loader.restore(sess,"/tmp/model/model")
    
    def save_agent(self):
        saver = tf.train.Saver()
        sess = get_session()
        saver.save(sess,"/tmp/model/model")
    
    def load_hyper_parameters(self):
        pass
    
    def save_hyper_parameters(self):
        pass
    
    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def sync_target(self,update='hard'):
        sess = get_session()
        if update=='hard':
            sess.run(self.target_init_updates)
        else:
            sess.run(self.target_soft_updates)
