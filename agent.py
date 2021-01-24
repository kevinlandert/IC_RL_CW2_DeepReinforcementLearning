import numpy as np
import torch
import random
import math
import collections


class Agent:
        
    
    # Function to initialise the agent
    def __init__(self,epsilon = 0.99995,lr = 0.001,discount = 0.95,freq = 50, log_stats = False, debug = False,priority = 0.8):
        
        
        self.episode_length = 1000
        self.num_steps_taken = 0
        self.step = 0
        self.state = None
        self.action = None
        self.episode_nr = 0

        
        #For early stopping
        self.stop_training = False
        self.reached_goal = False
        
        
        #Debugging and logging
        self.debug = debug  
        self.log_stats = log_stats
        
        #Log used for visualizations -> deactivate to keep lightweight
        if self.log_stats:
            self.setup_log()

        
        #Hyperparamters
        self.punish_wall = True
        self.reward_goal = True
        self.initial_exploring = 1000
        self.epsilon = epsilon
        self.target_update_frequency = freq
        self.discount_rate = discount
        self.lr = lr
        self.batch_size = 256
        self.alpha_start = priority
        self.beta_start = 1-priority
        
        N_s,N_l,NE_l,E_s,E_l,SE_l,S_s,S_l,SW_l,W_s,W_l,NW_l = self.get_directions()
        self.directions = [N_l,NE_l,E_l,SE_l,S_l,W_l]
        self.actions_cnt = len(self.directions)
        
        #Initiate the replay buffer
        self.buff = ReplayBuffer(maxlen = 25000)

        #Initiate Q-Network
        self.dqn = DQN(out_dim = self.actions_cnt,discount_rate = self.discount_rate,lr = self.lr)
     
    #Available directions that where tried
    def get_directions(self):
        
        N_s = [0,0.01]
        N_l = [0,0.02]
        NE_l = [0.014142135,0.014142135]
        E_s = [0.01,0]
        E_l = [0.02,0]
        SE_l = [0.014142135,-0.014142135]
        S_s = [0,-0.01]
        S_l = [0,-0.02]
        SW_l = [-0.014142135,-0.014142135]
        W_s = [-0.01,0]
        W_l = [-0.02,0]
        NW_l = [-0.014142135,0.014142135]
    
        return N_s,N_l,NE_l,E_s,E_l,SE_l,S_s,S_l,SW_l,W_s,W_l,NW_l
    
        
    #Returns parameters of agent -> for debugging
    def get_parameters(self):
        
        return self.epsilon,self.lr,self.discount_rate,self.target_update_frequency

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):        
             
        #If episode finished 
        if (self.step == self.episode_length):
            
            #Log first success for debuding
            if self.log_stats:
                if self.stop_training & (self.success_nr == -1):
                    self.success_nr = self.episode_nr
                    self.success_numsteps = self.num_steps_taken
            
            #Reset to training if greedy not working line 95
            if self.stop_training & (not self.reached_goal):
                if self.debug:
                    print("Greedy not working switching back to exploring")
                self.stop_training = False
            
        
            #At 100 episodes test greedy 
            if (self.episode_length == 100) & (self.reached_goal):
                if self.debug:
                    print("Testing greedy")    
                self.stop_training = True
            
            #Enable if we want to print per episode statistics
            if self.debug:
                self.print_debug_stats()
                
            #Compute statistics for plotting of loss and reward curves
            if self.log_stats:
                self.log_statistics()
            
            
            
            #Reduce episode length every 5 episodes after initial exploring
            if (self.num_steps_taken >= 5 * self.initial_exploring) & ((self.episode_nr % 5) == 0) & (self.num_steps_taken != 0):
                self.episode_length = max(self.episode_length -50,100)
    
            self.episode_reward = 0
            self.episode_nr += 1
            self.episode_loss = 0
            self.step = 0
            self.reached_goal = False
            return True
        
        else:
        
            return False

    #Printing statistics for every episode
    def print_debug_stats(self):
        
            print("Start Episode: {}".format(self.episode_nr))
            print("Max Episode length: {}".format(self.episode_length))
            print("Total Steps: {}".format(self.num_steps_taken))
            print("Epsilon: {}".format(max(0.05,(self.epsilon ** self.num_steps_taken))))
            print("Buffsize: {}".format(self.buff.size))
            print("Priority (alpha): {}".format(self.buff.alpha))
            print("Beta: {}".format(self.get_beta()))
     
    #Setup all log  fields
    def setup_log(self):
        
        self.episode_loss = 0
        self.episodes = []
        self.episode_losses = []
        self.losses = []
        self.episode_reward = 0
        self.episode_rewards = []
        self.rewards = []
        self.episode_lengths = []
        self.episode_steps = []
        self.success_nr = -1
        self.success_numsteps = -1
    
    #Logging all stats for visualizations
    def log_statistics(self):
        
            self.episode_losses.append(self.episode_loss)
            self.episodes.append(self.episode_nr)
            self.episode_rewards.append(self.episode_reward / self.step)  
            self.episode_lengths.append(self.episode_length)
            self.episode_steps.append(self.step)
        
        
        
    # Function to get the next action
    def get_next_action(self, state):
          
        #Forced initial Exploring phase
        if self.num_steps_taken < self.initial_exploring:
            direction = np.random.choice(len(self.directions))
        #Deterministic if switched to greedy
        elif self.stop_training:
            qvalue = self.dqn.get_q_value(state)[0]  
            direction = np.argmax(qvalue)
        #Modified epsilon greedy
        elif ((np.random.random() < max(0.05,1 - (self.epsilon ** self.num_steps_taken))) & ((self.step / self.episode_length) < 1 - (self.epsilon ** self.num_steps_taken))):
            qvalue = self.dqn.get_q_value(state)[0]  
            direction = np.argmax(qvalue)
        else:
            direction = np.random.choice(len(self.directions))  
            
             
        action = self.directions[direction]
        
        #Saving
        self.state = state
        self.action = direction 
        self.num_steps_taken += 1
        self.step += 1
        
        
        return action
    
    
    

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
                
        #Reward function shaping
        reward = math.exp(-5 * distance_to_goal)
            
        #Additional reward for reaching goal
        if (distance_to_goal < 0.03) & self.reward_goal :
            reward *= 1.2
        #Additional punishment for walking into wall
        elif (np.array_equal(self.state, next_state)) & self.punish_wall:
            reward /=1.2
            
        
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        
        #Stop saving transitions if just running greedy policy
        if not self.stop_training:
            self.buff.append(transition)
        loss = self.train_agent()  
        
        #Save statistics
        if self.log_stats:
            self.episode_reward += reward   
            self.rewards.append(reward)
            self.episode_loss += loss
            self.losses.append(loss)
            
        #For early stopping training
        if (self.episode_length == 100) & (distance_to_goal < 0.03):
            self.reached_goal = True
                
        

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        qvalue = self.dqn.get_q_value(state)[0]  
        direction = np.argmax(qvalue)
        action = self.directions[direction]
        
        return action
    
        
    
    #Sampling mini batches, training our qnetwork and updating weights of buffer and target network
    def train_agent(self):
        
        loss = 0
        
        #Wait until enough samples in buffer
        if(self.buff.size >= self.batch_size):
              
            #Alpha and Beta for PER buffer
            beta = self.get_beta()
            alpha = self.alpha_start * (self.epsilon ** self.num_steps_taken)
            
            #Keep training as long as not set to greedy
            if not self.stop_training:
                minibatch,weights = self.buff.sample(self.batch_size,beta,alpha)
                loss,priorities = self.dqn.train_q_network(minibatch,weights)
                #Update buffer weights
                self.buff.update_weights(priorities)
            
            #Update target network
            if(self.num_steps_taken % self.target_update_frequency) == 0:
                    self.dqn.update_target()
            
        return loss

    #Helper function to get beta of PER buffer
    def get_beta(self):
        
        total_steps = 50000  
        return min(1,self.beta_start + self.num_steps_taken * (1 - self.beta_start) / total_steps )   
        
        
#NN Class using Pytorch       
class Network(torch.nn.Module):
    
    
    def __init__(self,input_dimension,output_dimension):
        
        super(Network,self).__init__()
        nodes = 32
        
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=nodes)
        self.layer_2 = torch.nn.Linear(in_features=nodes, out_features=nodes)
        self.layer_3 = torch.nn.Linear(in_features = nodes, out_features = nodes)
        self.layer_4 = torch.nn.Linear(in_features = nodes, out_features = nodes)
        self.layer_5 = torch.nn.Linear(in_features = nodes, out_features = nodes)
        self.layer_6 = torch.nn.Linear(in_features = nodes, out_features = nodes)
        self.layer_7 = torch.nn.Linear(in_features = nodes, out_features = nodes)
        self.output_layer = torch.nn.Linear(in_features=nodes, out_features=output_dimension)
    
    
    
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_4_output))
        layer_6_output = torch.nn.functional.relu(self.layer_6(layer_5_output))
        layer_7_output = torch.nn.functional.relu(self.layer_7(layer_6_output))
        output = self.output_layer(layer_7_output)        
        return output
    
#Deep Q-Network Class
class DQN:
    
    def __init__(self,out_dim,discount_rate,lr):
        
        self.lr = lr
        self.discount_rate = discount_rate
        
        # Create a Q-network, and target network
        self.q_network = Network(input_dimension=2, output_dimension=out_dim)
        self.q_network_target = Network(input_dimension = 2, output_dimension = out_dim)


        # Using Adam Optimizer here
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.optimiser_target = torch.optim.Adam(self.q_network_target.parameters(),lr = self.lr)
        
        self.q_network_target.load_state_dict(self.q_network.state_dict())


        
    def train_q_network(self,mini_batch,weights):
        
        #Set all gradients in the optimizer to zero
        self.optimiser.zero_grad()
        
        #Compute loss for transition
        loss,priorities = self._calculate_loss(mini_batch,weights)
        
        #Compute gradients based on this loss. i.e. the gradients with respect to the Q-network parameters
        loss.backward()
        
        #Take one gradient step to update the Q-Network
        self.optimiser.step()
        
        #Returns loss as a scalar
        return loss.item(),priorities.numpy()
    
    #Computes the loss function
    def _calculate_loss(self,minibatch,weights):
        
        states,actions,rewards,next_states = minibatch
        
        #Create tensors
        states_tensor = torch.tensor(states,dtype = torch.float32)
        actions_tensor = torch.tensor(actions)
        rewards_tensor = torch.tensor(rewards,dtype = torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype = torch.float32)
        
        #Compute Q(s,a)
        q_states_prediction = self.q_network.forward(states_tensor)
        #Compute Q(s',a)
        q_next_states_prediction = self.q_network_target.forward(next_states_tensor).detach()
        
        #Double-Q-Learning
        #Get argmax of Q
        q_states_argmaxes = torch.argmax(q_states_prediction.detach(),dim = 1)
        #Get max of Q' based on this argmax
        q_next_states_maxes = q_next_states_prediction.gather(dim = 1,index = q_states_argmaxes.unsqueeze(-1)).squeeze(-1)
        
        #Get max across batch according to specification of CW
        state_action_q_values = q_states_prediction.gather(dim = 1,index = actions_tensor.unsqueeze(-1)).squeeze(-1)
        
        #R + gamma * Q(s',a)
        network_prediction = torch.add(rewards_tensor,q_next_states_maxes,alpha = self.discount_rate)

        
        #MSE Loss
        loss = (state_action_q_values - network_prediction.detach()).pow(2) * torch.tensor(weights,dtype = torch.float32)
        priorities = loss.detach() + 1e-4  #Add minimum priority for PER
        loss = loss.mean()  
        
        return loss,priorities
    
    #Gets Q-values for a particular state
    def get_q_value(self,state):
        
        input_tensor = torch.tensor(state,dtype = torch.float32)
        input_tensor = torch.unsqueeze(input_tensor,0)
        q = self.q_network.forward(input_tensor).detach().numpy()
        return q
    
    #Updates target network
    def update_target(self):
        
        self.q_network_target.load_state_dict(self.q_network.state_dict()) 
    
    
    
#Prioritized experience Replay Buffer Class
class ReplayBuffer:
    
        def __init__(self,maxlen,alpha = 0.8,beta = 0.2):
        
            self.maxlen = maxlen
            self.size = 0
            self.buffer = collections.deque(maxlen = maxlen)
            self.p = collections.deque(maxlen = maxlen)
            self.max_p = 0.05
            self.to_update_idx = None
            self.alpha = None
            self.beta = None
            
            
        #Appends an element to the buffer, removes old elements if full
        def append(self,element):
        
            if self.size == self.maxlen:
                self.buffer.popleft()
                self.p.popleft()
            self.buffer.append(element)
            self.p.append(self.max_p)
            self.size = min(self.size + 1,self.maxlen)
            
    
        #Samples transitions based on prioritization
        def sample(self,amount,beta,alpha):
            
            self.alpha = alpha
            self.beta = beta
            prob = (np.array(self.p) ** self.alpha)
            prob /= prob.sum()
            
            indexes = np.random.choice(np.arange(self.size),size = amount, replace = False, p = prob)
            samples =  [self.buffer[index]  for index in indexes]
            
            #Saves current batch for weight update
            self.to_update_idx = indexes
            
            #Compute weight of each sample
            weights = (self.size * prob[indexes]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype = np.float32)
            
            
            return zip(*samples), weights

        #Weight update for PER
        def update_weights(self,priorities):
            
            
            weights = np.abs(priorities)
            
            for i,id in enumerate(self.to_update_idx):
                self.p[id] = weights[i]
                
            self.max_p =  max(self.p)  
            self.to_update_idx = None
        
