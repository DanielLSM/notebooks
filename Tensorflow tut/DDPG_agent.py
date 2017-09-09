'''A complete DDPG agent, everything running on tensorflow should just run 
in this class for sanity and simplicity. Moreoever, every variable and
hyperparameter should be stored within the tensorflow graph to grant
increased performance'''
#agent=DDPG_agent(something,something)....
class DDPG_agent(object):
    #import tensorflow as tf
    def __init__(self,alpha):
        import tensorflow as tf
        self.alpha =tf.constant(alpha, name="alpha")
        #self.replay_memory = 
    def actor(self,input_observation):
        pass
    
    def critic(self):
        pass
    
    def target_actor(self):
        pass
    
    def target_critic(self):
        pass
    
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
    