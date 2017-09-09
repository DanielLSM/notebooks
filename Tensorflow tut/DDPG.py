'''A complete DDPG agent, everything running on tensorflow should just run 
in this class for sanity and simplicity. Moreoever, every variable and
hyperparameter should be stored within the tensorflow graph to grant
increased performance'''
#agent=DDPG_agent(something,something)....
import tensorflow as tf

class Actor(object):
    def __init__(self):
        #with tf.variable_scope("actor") as scope:
        #    x = tf.Variable()
        #return 1,2
        pass
class Critic(object):
    def __init__(self):
        #return 3,4
        pass



class DDPG_agent(object):
    
    def __init__(self,alpha,noise):
        #with tf.variable_scope("hyper_parameters"):
        #    self.alpha = tf.constant(alpha, name="alpha", dtype=tf.int32)
        #    self.noise = tf.Variable(noise, name="noise", dtype=tf.float32)
        self.alpha = alpha
        self.noise = noise 
        self.actor = Actor()
        self.critic = Critic()

        #self.actor, self.actor_target = Actor()
        #self.critic, self.critic_target = Critic()
        #self.replay_memory =
    def __call__(self,input_observation):
        #return action, q_value
        pass

    def __len__(self):
        #return memory_replay_size and/or number of episodes
        pass 
    
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
    
    def train(self):
        #batch something self.replay_memory.batch
        #feed some dict to a node on the graph
        pass
        
    def load_agent(self):
        pass
    
    def save_agent(self):
        pass
    
    def load_hyper_parameters(self):
        pass
    
    def save_hyper_parameters(self):
        pass
    
    def get_action(self,observation):
        #feed dict into actor, return action
        pass
    