# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:06:16 2020

@author: Soumya Srilekha
"""
### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:
	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""
########################################################
# # YOUR IMPLEMENTATION HERE
#Action value function has been defined here to increase code reusablity
#
def Q_value(policy, s, nA, value, gamma):
    # Initialzing Q_a
    Q_a = np.zeros(nA)
    #Iterate through all actions 
    for a in range(nA):
        #iterate through all the probabilities,next states, rewards, terminate for that action in that policy
        for probability , next_state, reward, terminate in policy[s][a]:
            #Calculate the action value function 
            Q_a[a] += probability * (reward + (gamma * value[next_state]))
            
    return Q_a
########################################################
    

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.
    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE 
    
    #Initialzing delta
    delta = np.ones(nS)
    #Iterate till the delta is smaller than the tolerance
    while max(delta) > tol:
        #for each state in all sates possible
        for s in range(nS):
            #initialize the state value function to zero.
            value = 0
            #iterate thorugh all actions for all states
            for a,a_probability in enumerate(policy[s]):
                #iterate through all the probabilities,next states, rewards, terminate for that action
                for probability , next_state, reward, terminate in P[s][a]:
                    #Calculate the state value function for all states 
                    value += (a_probability * probability * (reward + (gamma * value_function[next_state])))
            #update delta
            delta[s]= np.abs(value_function[s] - value)
            # update the current state value in the value function.
            value_function[s] = value

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.
    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
	############################
	# YOUR IMPLEMENTATION HERE #
    #--------------------------------------------------------------------------
    #Q Value has been defined in the beginning of the code to increase code reusablility
    #--------------------------------------------------------------------------
    #Policy Improvement Algorithm from pseudo-code
    #First initialize that the policy is not stable 
    #Iterate till stable policy is found
    p_s =0
    while not p_s:
        #Iterate through the states 
        for s in range(nS):
            #Find the best action from the current policy's state
            old_action = np.argmax(new_policy[s])
            #Finding the maximum of all actions
            pi_s = np.argmax(Q_value(P, s, nA, value_from_policy, gamma))
            # If old_action == pi(s) that means that the policy is stable and pi*is found
            #Update the new_policy
            new_policy[s] = np.eye(nA)[pi_s]
            #The policy wouldbe stable if this is satisfies
            if old_action == pi_s:
                p_s =1

	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.
    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
    temp = 1
    while temp:
        #Policy Evaluation
        V = policy_evaluation(P, nS, nA, new_policy, gamma=0.9, tol=1e-8)
        #Policy improvement
        new_policy = policy_improvement(P, nS, nA, V, gamma=0.9)
        #Check if the policies have converged 
        if (new_policy == policy).all():
            temp= 0
        #else Update the policy
        policy = new_policy.copy()
        
	############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.
    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    #--------------------------------------------------------------------------
    #Q Value has been defined in the beginning of the code to increase code reusablility
    #--------------------------------------------------------------------------
    #Initialzing delta
    delta = np.ones(nS)
    # Finding the value Function
    while max(delta) > tol:
        #for each state in all sates possible
        for s in range(nS):
            #Store all the previous state values
            V_s = V_new[s]
            #Find Vnew
            V_new[s] = max(Q_value(P,s, nA, V_new, gamma))
            #Update delta
            delta[s] = np.abs(V_new[s] - V_s)
    # update the policy with the newly found V_new
    policy_new = policy_improvement(P, nS, nA, V_new, gamma=0.9)
    
    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.
    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ----------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for episode in range(n_episodes):
        print("Episode number : ",episode+1)
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            #in order to fetch the total reward wthe best action has to be selected 
            action = np.argmax(policy[ob, :])
            ob, reward, done, info = env.step(action)
            total_rewards += reward
            
            ##########################
    return total_rewards