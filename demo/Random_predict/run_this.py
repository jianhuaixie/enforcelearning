from random_env import RandomEnv
# from RL_brain_sarsalamda import RL
from RL_brain import DeepQNetwork

def run_maze(env,RL):
    step = 0

    for episode in range(100000):
        # inital observation
        pred_right = 0
        pred_error = 0
        for i in range(100):
        # while True:
            # fresh env
            # env.render()
            observation = env.reset()
            # RL choose action based on observaation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_,reward,done = env.step(action,observation)
            if(reward==1):
                pred_right += 1
            else:
                pred_error += 1
            # print(action,observation_[-1])
            RL.store_transition(observation,action,reward,observation_)

            # if(step>200) and (step%5==0):
            RL.learn()
                # RL.learn(observation,action,reward,observation_,observation_[-1])

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            # if done:
            #     break
            step += 1
        accuracy = pred_right / (pred_error + pred_right)
        print(accuracy)
    # end of game
    print('game over')
    print('total step:',step)
    # env.destroy()

if __name__ == "__main__":
    env = RandomEnv()
    RL = DeepQNetwork(env.n_actions,env.n_features,learning_rate=0.2,reward_decay=0.9,
                      e_greedy=0.9,replace_target_iter=2000,memory_size=3000000,output_graph=False)
    # RL = RL(action_space=env.action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9)
    run_maze(env,RL)
    RL.plot_cost()