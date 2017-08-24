import numpy as np
import pandas as pd
import time

np.random.seed(2) # reproducible

N_STATE = 6  # the length of the 1 dimentional word
ACTIONS = ['left','right']   # avialable actions
EPSILON = 0.9  # greedy policy
ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor
MAX_EPISONDES = 13  # maximum episodes
FRESH_TIME = 0.3  # frech time for one move

def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    print(table)
    return table

def choose_action(state,q_table):
    # this is how to choose an action
    state_actions = q_table.iloc[state,:]
    if(np.random.uniform()>EPSILON) or (state_actions.all() == 0): # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S,A):
    # this is how agent will interact with the enviroment
    if A == 'right': # move right
        if S == N_STATE - 2: # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S-1
    return S_,R

def update_env(S,episode,step_conter):
    # this is how enviroment be updated
    env_list = ['-']*(N_STATE-1)+['T']  # '-------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s:total_steps = %s' % (episode+1,step_conter)
        print ('\r{}'.format(interaction))
        time.sleep(2)
        print('\r                            ')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATE,ACTIONS)
    for episode in range(MAX_EPISONDES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,episode,step_counter)
        while not is_terminated:
            A = choose_action(S,q_table)
            S_,R = get_env_feedback(S,A)  # take action & get next state and reward
            q_predict = q_table.ix[S,A]
            if S_ != 'terminal':
                q_target = R + GAMMA*q_table.iloc[S_,:].max()  # next staate is not terminal
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # termiaate this episode

            q_table.ix[S,A] += ALPHA*(q_target-q_predict)  # update
            S = S_  # move to next state

            update_env(S,episode,step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)




















