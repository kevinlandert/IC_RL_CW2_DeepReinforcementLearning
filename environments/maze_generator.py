import sys, getopt

import time
import numpy as np

from random_environment import Environment
from agent import Agent
from matplotlib import pyplot as plt

    
def main_gridsearch(argv):
    
    random_seed = 0
    
    opts,args = getopt.getopt(argv,"s:")
    
    for opt,arg in opts:
        
        if opt == "-s":
            random_seed = arg
    
    outfile = "results/results_" + str(random_seed) +".txt"
    
    with open(outfile, 'a+') as file:
        
        print("STARTING TEST FOR SEED: " + str(random_seed),file = file)
    
    epsilons = [0.9999,0.99991,0.99992]
    lrs = [0.001,0.002,0.003]
    discounts = [0.9,0.95]
    freq = [50,100,150]
    print("Seed;Epsilon;Lr;Discount;Updata_freq;Run;Success;Steps;Num_Steps",file = file)
    for a in epsilons:
        for b in lrs:
            for c in discounts:
                for d in freq:
                    for run in range(3):
                        
                        result,distance,num_steps = run_test(random_seed,a,b,c,d,run)
                        
                        with open(outfile, 'a+') as file:
                            print("{};{};{};{};{};{};{};{};{}".format(random_seed,a,b,c,d,run,result,distance,num_steps),file = file)
                        
                        
    file.close()
    
    
    
    
def main_testagent():
    
    
    #random_seed = int(time.time())
    random_seed = 0
    
    
    
    for i in range(100):
              
        #random_seed = seeds[i]
        a = 0.999925 #old 0.99991
        b = 0.001
        c = 0.95
        d = 50
        
        for run in range(1):
            
            result,distance,num_steps = run_test(random_seed,a,b,c,d,run)
            
        random_seed += 1  
    
def run_test(random_seed,a,b,c,d,run,draw = False):

        
    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True
    draw_greedy_start = 'none'

    # Create a random seed, which will define the environment
    np.random.seed(int(random_seed))

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent(a,b,c,d,log_stats = False,debug=False)
    agent.set_seed(random_seed)

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600 * (2.8 if draw_greedy_start == 'full' else 1)

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        new = agent.has_finished_episode()
        if new:
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on:
            #environment.show(state,True,agent,new)
            environment.show(state,draw_greedy_start,agent,new,random_seed)
        break

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

        
        
    #environment.draw_greedy(environment.init_state,agent,success)
    
    num_steps_taken = agent.num_steps_taken
    
    if draw:
                
        losses = agent.losses
        rewards = agent.rewards
        episode_losses = agent.episode_losses
        episode_rewards = agent.episode_rewards
        episodes = agent.episodes
        episode_lengths = agent.episode_lengths
        episode_steps = agent.episode_steps
        success_nr = agent.success_nr if agent.success_nr != -1 else agent.episode_nr
        success_numsteps = agent.success_numsteps if agent.success_numsteps != - 1 else num_steps_taken
        
        
        
        #Draw loss and reward per step
        fig = plt.figure(figsize=(16, 8))
        draw_cnt = min(num_steps_taken,40000)
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(np.arange(draw_cnt),rewards[:draw_cnt],label = 'Stepwise Rewards',color = 'red')
        ax2 = ax1.twinx()
        ax2.plot(np.arange(draw_cnt),losses[:draw_cnt],label = 'Stepwise Losses',color = 'blue')
        plt.title(num_steps_taken)
        plt.legend()
        fig.savefig("results/stepwise_loss_and_rewards_" + str(random_seed) + "_" + str(run) + ".png")
        plt.close(fig)
        
        #Draw loss and reward per episode
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(episodes[:success_nr],episode_rewards[:success_nr],label = 'Per Episodes Rewards',color = 'red')
        ax2 = ax1.twinx()
        ax2.plot(episodes[:success_nr],episode_losses[:success_nr],label = 'Per Episode Losses',color = 'blue')
        plt.title(num_steps_taken)
        plt.legend()
        fig.savefig("results/episode_loss_and_rewards_" + str(random_seed) + "_" + str(run) + ".png")
        plt.close(fig)
        
        #Draw episode length development
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(episodes[:success_nr],episode_lengths[:success_nr],label = 'Episode Lengths',color = 'red')
        ax2 = ax1.twinx()
        ax2.plot(episodes[:success_nr],episode_steps[:success_nr],label = 'Steps per Episode taken',color = 'blue')
        plt.title('yes' if has_reached_goal else "no")
        plt.legend()
        fig.savefig("results/episode_lengths_steps_" + str(random_seed) + "_" + str(run) + ".png")
        plt.close(fig)
    
    
    
    
        #Draw greedy policy
        counter = str(random_seed) + "_" + str(run)
        environment.draw_greedy(environment.init_state,agent,counter)
    
    
    
    
    if has_reached_goal:
        return 'yes', str(step_num),str(num_steps_taken)
    else:
        return 'no',str(distance_to_goal),str(num_steps_taken)

    
    
# Main entry point
if __name__ == "__main__":

    #main_gridsearch(sys.argv[1:])
    main_testagent()