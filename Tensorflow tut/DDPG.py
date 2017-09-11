'''A complete DDPG agent, everything running on tensorflow should just run 
in this class for sanity and simplicity. Moreoever, every variable and
hyperparameter should be stored within the tensorflow graph to grant
increased performance'''
#agent=DDPG_agent(something,something)....
from canton.misc import get_session
import tensorflow as tf
from copy import copy
from baselines.ddpg.memoryNIPS import Memory
import tensorflow.contrib as tc
from models import *

class DDPG_agent(object):
    
    def __init__(self,observation_dims=41, action_dims=18,
        alpha=0.9,gamma=0.99, noise=0.2,memory_size=1000,batch_size=32,tau=1.,
        actor_l2_reg=1e-4,critic_l2_reg=1e-4,train_multiplier=1):
        observation_shape = (None,observation_dims)
        action_shape = (None,action_dims)

        #Inputs
        self.observation = tf.placeholder(tf.float32, shape=observation_shape, name='observation')
        self.action = tf.placeholder(tf.float32, shape=action_shape, name='action')        
        self.observation_after = tf.placeholder(tf.float32, shape=observation_shape, name='observation_after')
        self.reward = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')        
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        
        #self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        #self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
 
        #Hyper Parameters
        self.alpha = alpha
        self.noise = noise
        self.gamma = gamma
        self.tau = tau
        self.actor_l2_reg = actor_l2_reg
        self.critic_l2_reg = critic_l2_reg
        self.batch_size = batch_size
        self.train_multiplier = train_multiplier

        #Replay Memory
        self.memory_replay = Memory(limit=memory_size,action_shape=(action_dims,),observation_shape=(observation_dims,))  
        
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
        self.critic_loss = tf.reduce_mean((self.q1_target - self.q1_predict)**2)

        # Actor Nodes
        self.a1_predict = self.actor(self.observation)
        self.q1_predict = self.critic(self.observation,self.a1_predict,reuse=True)
        self.actor_loss = tf.reduce_mean(- self.q1_predict) 

        # Infer
        self.a_infer = self.actor(self.observation,reuse=True)
        self.q_infer = self.critic(self.observation,self.a_infer,reuse=True)


        # Setting Nodes to Sync target networks
        self.setup_target_network_updates()


        # Train Boosters

        # Optimzers
        self.opt_actor = tf.train.AdamOptimizer(1e-4)
        self.opt_critic = tf.train.AdamOptimizer(3e-4)

        # L2 weight loss
        #decay_c = tf.reduce_sum([tf.reduce_sum(w**2) for w in cw])* 1e-7
        #decay_a = tf.reduce_sum([tf.reduce_sum(w**2) for w in aw])* 1e-7
        critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            #for var in critic_reg_vars:
                #logger.info('  regularizing: {}'.format(var.name))
            #logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
        self.critic_reg = tc.layers.apply_regularization(
            tc.layers.l2_regularizer(self.critic_l2_reg),
            weights_list=critic_reg_vars
        )

        actor_reg_vars = [var for var in self.actor.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
        self.actor_reg = tc.layers.apply_regularization(
            tc.layers.l2_regularizer(self.actor_l2_reg),
            weights_list=actor_reg_vars
        )    
        #self.decay_c = 0
        #self.decay_a = 0

        # Nodes to run one backprop step on the actor and critic
        self.cstep = self.opt_critic.minimize(self.critic_loss+self.critic_reg,
            var_list=self.critic.trainable_vars)
        self.astep = self.opt_actor.minimize(self.actor_loss+self.actor_reg,
            var_list=self.actor.trainable_vars)

        # Initialize and Sync Networks
        self.initialize()
        self.sync_target()        
        print('agent initialized :>')
    
    def __call__(self,input_observation):
        feed_dict = {self.observation: input_observation}
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
        return self.memory_replay.nb_entries
         
    def initialize(self):
        sess = get_session()
        sess.run(tf.global_variables_initializer())
        #self.actor_optimizer.sync()
        #self.critic_optimizer.sync()
        #self.sess.run(self.target_init_updates)

    def feed_experience(self,obs0, action, reward, obs1, terminal1):
        #it is thread safe
        self.memory_replay.append(obs0, action, reward, obs1, terminal1)

    def train(self):
        batch = self.memory_replay.sample(self.batch_size)
        batch_size = self.batch_size

        if len(self) > batch_size:

            for i in range(self.train_multiplier):

                sess = get_session()
                res = sess.run([self.critic_loss,
                    self.actor_loss,
                    self.cstep,
                    self.astep],
                    feed_dict={
                    self.observation:batch['obs0'],
                    self.action:batch['actions'],
                    self.observation_after:batch['obs1'],
                    self.reward:batch['rewards'],
                    self.terminals1:batch['terminals_1']})
                print('closs: {:6.6f} aloss: {:6.6f}'.format(
                    res[0],res[1]),end='\r')
        return res


        
    def load_agent(self):
        loader = tf.train.Saver()
        sess = get_session()
        loader.restore(sess,"/tmp/model/model")
    
    def save_agent(self):
        saver = tf.train.Saver()
        sess = get_session()
        saver.save(sess,"/tmp/model/model")
    
    #def load_hyper_parameters(self):
        #pass
    
    #def save_hyper_parameters(self):
        #pass
    
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
