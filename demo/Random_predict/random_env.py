import numpy as np

np.random.seed(1)

class RandomEnv(object):
    action_dim = 1
    state_dim = 100  # 状态就是之前100个数
    def __init__(self):
        self.action_space = [0,1,2,3,4,5,6,7,8,9]
        self.n_actions = len(self.action_space)
        self.n_features = self.state_dim

    def reset(self):
        observation = []
        for i in range(self.state_dim):
            observation.append(np.random.randint(0,10))
        return np.array(observation)

    def step(self,action,observation):
        pred_action = np.random.randint(0,10)
        reward = -1
        done = False
        if(action==pred_action):
            reward = 1
            done = True
        observation_ = np.append(observation[1:],pred_action)
        return observation_,reward,done

if __name__ == '__main__':
    env = RandomEnv()
    observation = env.reset()

    o_,r,d = env.step(5,observation)
    print(o_)
    print(o_[-1])