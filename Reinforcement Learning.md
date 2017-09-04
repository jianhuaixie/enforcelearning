# <center>Reinforcement Learning</center>

# --------------------------------------------------------------------
### <center>tabular Q learning</center>
![](https://morvanzhou.github.io/static/results/rl/2-1-1.png)

- 所有的Q values(行为值)放在q_table中，更新q_table也是在更新机器人的行为准则
- q_table的index是所有对应的state（机器人位置）
- columns是对应的action（机器人行为）

- 整个算法就是不断更新Q table里的值，然后再根据新的值来判断要在某个state采取怎样的action
- Q-Learning算是一个off-policy的算法，


# --------------------------------------------------------------------
### <center>Sarsa</center>
![](https://morvanzhou.github.io/static/results/rl/3-1-1.png)

- 他在当前 state 已经想好了 state 对应的 action, 而且想好了 下一个 state_ 和下一个 action_ (Qlearning 还没有想好下一个 action_)
- 更新 Q(s,a) 的时候基于的是下一个 Q(s_, a_) (Qlearning 是基于 maxQ(s_))

# --------------------------------------------------------------------
### <center>Sarsa-lambda</center>
![](https://morvanzhou.github.io/static/results/rl/3-3-1.png)

- Q-Learning和Sarsa都只更新获取到reward的前一步，而Sarsa-lambda就是更新获取到reward的前lambda步，lambda取值[0,1]之间。
- 如果 lambda = 0, Sarsa-lambda 就是 Sarsa, 只更新获取到 reward 前经历的最后一步.
- 如果 lambda = 1, Sarsa-lambda 更新的是 获取到 reward 前所有经历的步.

# --------------------------------------------------------------------
### <center>Deep Q Network</center>
![](https://morvanzhou.github.io/static/results/rl/4-1-1.JPG)

- 记忆库 (用于重复学习)
- 神经网络计算 Q 值
- 暂时冻结 q_target 参数 (切断相关性)
- 两个神经网络target_net,eval_net,前者预测q_target，后者预测q_eval，前者不会及时更新参数，后者及时更新参数。
- eval_net是不断被提升的，是一个可以被训练的网络trainable=True. 而 target_net 的 trainable=False.
- DQN记录所有经历过的步，是一种off-policy方法。

# --------------------------------------------------------------------
### <center>Double DQN</center>
![](https://morvanzhou.github.io/static/results/rl/4-5-2.png)

- DQN基于Q-Learning，Qmax会导致Q现实当中的过估计(overestimate)，Double DQN为解决这一问题。
- 有两个神经网络: Q_eval (Q估计中的), Q_next (Q现实中的).
- 原本的 Q_next = max(Q_next(s', a_all)).
- Double DQN 中的 Q_next = Q_next(s', argmax(Q_eval(s', a_all))). 


# --------------------------------------------------------------------
### <center>Prioritized Experience Replay (DQN)</center>
![](https://morvanzhou.github.io/static/results/rl/4-6-1.png)

- batch抽样的时候并不是随机抽样，而是按照Memory中的样本优先级来抽。
- TD-error，也就是Q现实-Q估计来规定优先学习的程度。
- 有了TD-error就有了优先级p，为了性能，并不对得到的样本进行排序，采用SumTree算法。


# --------------------------------------------------------------------
### <center>Dueling DQN</center>
![](https://morvanzhou.github.io/static/results/rl/4-7-2.png)

![](https://morvanzhou.github.io/static/results/rl/4-7-4.png)

- 将每个动作的Q拆分成state的Value加上每个动作的Advantage。
- DQN神经网络直接输出的是每种动作的Q值，而Dueling DQN每个动作的Q值是两部分组成。


# --------------------------------------------------------------------
### <center>Policy Gradients</center>
![](https://morvanzhou.github.io/static/results/rl/5-1-1.png)

- 不同于Q-Learning和Sarsa，接收环境信息（observation），选择action，得到最大的value。policy gradient跳过了value这个阶段，直接输出所有动作的概率分布。
- 上图是基于整条回合数据的更新，也叫REINFORCE方法。
- log(Policy(s,a))*V 中的 log(Policy(s,a)) 表示在 状态 s 对所选动作 a 的吃惊度, 如果 Policy(s,a) 概率越小, 反向的 log(Policy(s,a)) (即 -log(P)) 反而越大. 
- 比如，一个很小的Policy(s,a)，意味着不常选的动作，拿到一个大的R，也就是大的V，那么 -log(Policy(s, a))*V 就更大，表示更加惊讶，需要大幅度调参。

# --------------------------------------------------------------------
### <center>Actor Critic</center>
![](https://morvanzhou.github.io/static/results/rl/6-1-1.png)

- 结合了 Policy Gradient (Actor) 和 Function Approximation (Critic) 的方法. Actor 基于概率选行为, Critic 基于 Actor 的行为评判行为的得分, Actor 根据 Critic 的评分修改选行为的概率.

- 相比于REINFORCE方法这种回合更新，可以单步更新，比传统的Policy Gradient要快。
- Critic难收敛，加上Actor的更新，Critic更难收敛。DDPG解决这些问题。
- Actor 在运用 Policy Gradient 的方法进行 Gradient ascent 的时候, 由 Critic 来告诉他, 这次的 Gradient ascent 是不是一次正确的 ascent, 如果这次的得分不好, 那么就不要 ascent 那么多.
- Actor 想要最大化期望的 reward, 在 Actor Critic 算法中, 我们用 “比平时好多少” (TD error) 来当做 reward。
- Critic 的更新很简单, 就是像 Q learning 那样更新现实和估计的误差 (TD error) 就好了。只是一个神经网络。

	
		s = env.reset()
	    t = 0
	    track_r = []
	    while True:
	        if RENDER: env.render()
	        a = actor.choose_action(s)
	        s_, r, done, info = env.step(a)
	        if done: r = -20
	        track_r.append(r)
	        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
	        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
	        s = s_

 - 上面是Actor-Critic的核心代码，第一步就是初始化得到s
 - 第二步是Actor根据s选择一个动作a
 - 第三步是环境根据选择的动作a更新得到回报r和下一次s_,当然是否结束的信息
 - 第四步，判断是否结束，将s，r，s_作为输入传给Critic，得到一个比平时好多少的td_error衡量指标
 - 第五步，将s和a以及td_error给Actor训练，更新其参数
 - 第六步，更新环境状况s，开启新的一轮学习

# --------------------------------------------------------------------
### <center>Deep Deterministic Policy Gradient (DDPG)</center>
![](https://morvanzhou.github.io/static/results/rl/6-2-0.png)

- 上图是Actor更新参数部分，前半部分grad[Q]是从Critic来的，就是Critic告诉Actor，要怎么移动，才能最大Q。
- 后面部分grad[μ]是从Actor来的，Actor要修改自身参数，最大化Q。

![](https://morvanzhou.github.io/static/results/rl/6-2-1.png)

- 上图是Critic参数更新部分，借鉴了DQN和Double DQN的方式，有两个计算Q的神经网络。
- Q_target中依据下一状态，用Actor来选择动作，而这时的Actor也是一个Actor_target（有着很久以前的参数），使用这种方法获得的Q_target能像DQN那样切断相关性，提高收敛性。

# --------------------------------------------------------------------
##### <center> DDPG神经网络图</center>
![](https://morvanzhou.github.io/static/results/rl/6-2-2.png)

- 使用Actor Critic的结构，但输出的不是行为的概率，而是具体的行为，用于连续动作（continuous action）的预测，DDPG结合了DQN结构，提高了Actor Critic的稳定性和收敛性。

		actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
		critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
		actor.add_grad_to_graph(critic.a_grads)

		for i in range(MAX_EPISODES):
    	s = env.reset()
    	ep_reward = 0
	    for j in range(MAX_EP_STEPS):
	        if RENDER:
	            env.render()
	        # Add exploration noise
	        a = actor.choose_action(s)
	        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
	        s_, r, done, info = env.step(a)
	        M.store_transition(s, a, r / 10, s_)
	        if M.pointer > MEMORY_CAPACITY:
	            var *= .9995    # decay the action randomness
	            b_M = M.sample(BATCH_SIZE)
	            b_s = b_M[:, :state_dim]
	            b_a = b_M[:, state_dim: state_dim + action_dim]
	            b_r = b_M[:, -state_dim - 1: -state_dim]
	            b_s_ = b_M[:, -state_dim:]
	
	            critic.learn(b_s, b_a, b_r, b_s_)
	            actor.learn(b_s)
	        s = s_

- 上面是DDPG的核心代码，第一步是初始化得到环境状态s
- 第二步是Actor根据s选择动作a，并且增加一些探索
- 第三步，根据动作a更新环境状态s_，得到回报r，以及是否任务完成
- 第四步，存储s,a,r,s_（Actor之前的状态，切断相关性）,然后从记忆中拿出b_s,b_a,b_r,b_s_给Critic学习，更新其eval_net和target_net参数，得到一个衡量Actor所选动作的好坏程度的a_grads衡量指标
	- 如何得到a_grads：Critic有一个可训练的eval_net和不可训练的target_net
	- target_q=r+gamma*q_,q是从eval_net得到，q_是从target_net得到
	- TD_error，也就是loss=tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
	- 然后C_train就梯度下降这个loss更新参数，a_grads=tf.gradients(self.q, a)[0]）
- 第五步，将s和a以及a_grads给Actor训练，更新其eval_net和target_net参数
	- Actor是如何利用a_grads的：Actor有一个可训练的eval_net和一个不可训练的target_net
	- a是eval_net得到，a_是target_net得到
	- 从Critic拿到的a_grads和a进行gradients得到policy_grads，也就是选动作的一个策略梯度，然后学习这个梯度，更新eval_net的参数
	- 其中target_net得到的a_要给Critic评判动作好不好来用
- 第六步，更新环境状况s，开启新的一轮学习

# --------------------------------------------------------------------
### <center>Asynchronous Advantage Actor-Critic (A3C)</center>
![](https://morvanzhou.github.io/static/results/rl/6-3-1.png)

A3C的算法实际上就是将Actor-Critic放在了多个线程中进行同步训练，多个Worker工作，然后将学到的经验同步共享到一个中央大脑。中央大脑最怕一个人的连续性更新，比如DQN使用记忆库来更新，打乱了经历间的相关性，如果中央大脑只有一个人在更新，所有的经历都是相关的，如果有多个Worker一起更新，更新用的经历是不相关的，达到了和DQN一样的目的。

中央大脑有global_net和他的参数，每个Worker有global_net的副本local_net，可以定时向global_net推送更新，然后定时从global_net那获取综合版的更新。

- 使用Normal distribution来选择动作，在搭建神经网络的时候，Actor要输出动作的均值mu和方差sigma，然后放到Normal distribution去选择动作。
- 计算Actor loss的时候需要使用Critic提供的TD error作为gradient ascent的导向。
- Critic，只需要得到他对于state的价值就好了，用于计算TD error。
	
		with tf.device("/cpu:0"):
	        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
	        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
	        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
	        workers = []
	        # Create worker
	        for i in range(N_WORKERS):
	            i_name = 'W_%i' % i   # worker name
	            workers.append(Worker(i_name, GLOBAL_AC))
	
	    COORD = tf.train.Coordinator()
	    SESS.run(tf.global_variables_initializer())

	    worker_threads = []
	    for worker in workers:
	        job = lambda: worker.work()
	        t = threading.Thread(target=job)
	        t.start()
	        worker_threads.append(t)
	    COORD.join(worker_threads)

- 上面是A3C的核心代码之一，第一步，构建两个优化器OPT_A和OPT_C
- 第二步，构建Global AC
- 第三步，创建Worker，添加进workers队列，每个Worker都共享上面创建的Global AC
- 第四步，创建Tensorflow用于并行的工具COORD
- 第五步，添加一个工作线程，并让tf的线程调度

	- 看看worker是如何work的？第一步是创建自己的环境，自己的名字，自己的本地网络AC，并绑定上共享过来的globalAC
	- 第二步，初始化环境s
	- 第二步，AC根据环境s选择动作a，得到下一步环境s_,回报r以及是否结束done
	- 第三步，添加s，a，r进入缓存，判断是否到了更新到中央大脑的步骤，进行sync操作
	- 第四步，计算得到TD_error
		- 看看如何得到TD_error，首先要计算下一state的value也就是s_v_
		- 然后从buffer_r中进行n_steps forward view，v_s_=r+Gamma*v_s_，然后添加到下state value的缓存队列中buffer_v_target
		
	- 第五步,将缓存中的数据推送更新到globalAC，然后清空缓存，获取globalAC的最新参数

		- 其中Worker在update_global的时候，就是将本地的梯度应用到中央大脑
			 
				SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
				===》
				self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params)) #优化器对a_grads进行梯度下降且更新Actor参数
                self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params)) #优化器对c_grads进行梯度下降且更新Critic参数
				===》
				self.a_grads = tf.gradients(self.a_loss, self.a_params) #a_grads是从a_loss求梯度而来
                self.c_grads = tf.gradients(self.c_loss, self.c_params) #c_grads是从c_loss求梯度而来
				===》
				self.a_loss = tf.reduce_mean(-self.exp_v) #a_loss是Actor所选的动作的期望价值均方差
				self.c_loss = tf.reduce_mean(tf.square(td))
				===》
				exp_v = log_prob * td  # Actor所选动作的期望价值是由两部分乘积，一部分是自己所做动作的均值mu和方差sigma放到Normal distribution得到的log_prob，另一部分是从Critic得来的td_error
				log_prob = normal_dist.log_prob(self.a_his) # Actor根据动作a_his，得出动作的价值
				td = tf.subtract(self.v_target, self.v, name='TD_error')      
				===》  # Actor所选动作自身的价值和Critic的价值v之间的差值
				normal_dist = tf.contrib.distributions.Normal(mu, sigma)
				mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope) # Actor那获得均值mu和方差sigma，Critic获得状态价值v
				
	- 第六步，更新环境，进行下一轮学习
		
		

# --------------------------------------------------------------------
### <center>Distributed Proximal Policy Optimization(DPPO)</center>
![](https://morvanzhou.github.io/static/results/rl/6-4-2.png)
	
解决Policy Gradient不好确定Learning rate（或者step size）的问题，因为如果step size过大，学出来的policy会一直乱动，不会收敛，但如果step size太小，对于完成训练，我们会等到绝望。PPO利用new Policy和old Policy的比例，限制了new Policy的更新幅度，让Policy Gradient对稍微大点的step size不那么敏感。

PPO是基于Actor-Critic算法，Actor想最大化J_PPO，Critic想最小化L_BL，Critic的loss好说，就是减少TD error，而Actor的就是在old Policy上根据Advantage（TD error）修改new Policy，advantage大的时候，修改幅度大，让new Policy更可能发生，而且附加一个KL Penalty（惩罚项），如果new Polic和old Policy差太多，那么KL divergence就越大，通俗来说，这个优势能能让新的策略大幅度修改，但也不要过头，这样会导致矫枉过正以致一条道走到黑，也就是保守主义的思想，不要让new Policy比old Policy差太多，如果差太多，相当于用了一个大的Learning rate，比较难收敛。

![](https://morvanzhou.github.io/static/results/rl/6-4-3.png)

pi是Actor，PPO更新Actor和Critic的时候，将pi的参数复制给oldpi，就是上面那个update_oldpi这个operation做的事情，Actor使用normal distribution正太分布输出动作。
更新Critic的时候，根据计算出来的discounted_r和自己神经网络分析出来的state value之间的差（advantage），然后最小化这个差值。
discounted_r是一个episode不断step积累下来的reward
更新Actor的时候，有两种方式，一种是KL penalty来更新，

![](https://morvanzhou.github.io/static/results/rl/6-4-4.png)

还有一种是clipped surrogate objective，就是限制new Policy的变化幅度，和KL penalty的规则差不多。
![](https://morvanzhou.github.io/static/results/rl/6-4-6.png)

	for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)


- 上面是PPO的主循环代码，第一步，初始化环境得到s
- 第二步，ppo的Actor根据s选择动作a，Actor使用normal distribution正太分布输出动作。
- 第三步，环境根据a更新得到s_,r,以及是否结束done
- 第四步，将s_,r,a加入到buffer，
- 第五步，如果到了需要更新ppo的步骤，更新ppo

	- 从s_，Critic获得价值v_s_
	- 将buffer中的r都加上一个GAMMA*v_s_得到一个discounted_r的列表
	- 用缓存中的s，a，以及上面的discounted_r用于ppo的更新

		- Actor，update_oldpi 就是不断将oldpi里的参数更新pi中的
		- Critic，输入s和r得到advantage， self.advantage = self.tfdc_r - self.v
		- atrain，就是最小化aloss，self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))，需要求一个surr和一个惩罚项kl
		surr = ratio*self.tfadv  ,tfadv是Actor的advatange
		ratio = pi.prob(self.tfa)/oldpi.prob(self.tfa)  ，tfa是Actor的action
		- ctrain,update Critic,输入s和r，就是最小化closs， self.closs = tf.reduce_mean(tf.square(self.advantage))
		 self.advantage = self.tfdc_r - self.v  ，tfdc_r是Critic的discounted_r，v是Critic的神经网络根据s和a得到的价值


DPPO就是借鉴A3C的并行方法，将PPO各个on-policy学习到的经验分享到Global PPO。

	- 用OpenAI提出的Clipped Surrogate Objective
	- 使用多个线程（workers）平行在不同的环境中收集数据
	- workers共享一个Global PPO
	- workers不会自己算PPO的gradients，不会像A3C那样推送Gradients给Global net
	- workers只推送自己采集的数据给Global PPO
	- Global PPO拿到多个workers一定批量的数据后进行更新（更新时worker停止采集）
	- 更新完后，workers用最新的Policy采集数据



