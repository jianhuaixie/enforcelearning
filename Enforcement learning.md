# <center>Enforcement learning</center>

## 强化学习分类和基本组成

	在第t步agent的工作流程是执行一个动作At，获得该动作之后的环境观测状况Qt，以及获得这个动作的反馈奖赏Rt。然后不断累积反馈奖赏且最大化。

	那么动作A就是所有股票的买卖组成的向量。
	状况Q就是动作A发生后所有股票的描述状态。
	R就是动作A执行后总金额与上一动作后的总金额差值。

	机器人agent的组成：
	由三部分组成Policy，Value function和Model，并不是必须同时存在。
	Policy，根据当前的observation来决定action。从state到action的映射，在某种状态下执行某个动作的概率。
	Value function，预测当前状态下未来可能获得的reward的期望，用来衡量当前状态的好坏。
	Model，预测environment下一步会做出什么样的改变，从而预测agent接收到的状态或者reward是什么。

		-通过价值选行为：Q-Learning，Sarsa，Deep Q Network
		-直接选行为：Policy Gradients
		-想象环境并从中学习：Model based RL

##### Model-free和Model-based

![](https://morvanzhou.github.io/static/results/ML_intro/RLmtd1.png)

采取model-based RL，机器人通过过往经验，先理解真实世界，并建立一个模型来模拟现实世界的反馈，最后不仅可以在现实世界中玩耍，还可以在模拟的世界中玩耍。

Model-free方法有很多，Q learning，Sarsa，Policy Gradients。而model-based RL只是多了一道程序，为真实世界建模，也可以说他们都是 model-free 的强化学习, 只是 model-based 多出了一个虚拟环境, 我们不仅可以像 model-free 那样在现实中玩耍,还能在游戏中玩耍, 而玩耍的方式也都是 model-free 中那些玩耍方式, 最终 model-based 还有一个杀手锏是 model-free 超级羡慕的. 那就是想象力.

Model-free 中, 机器人只能按部就班, 一步一步等待真实世界的反馈, 再根据反馈采取下一步行动. 而 model-based, 他能通过想象来预判断接下来将要发生的所有情况. 然后选择这些想象情况中最好的那种. 并依据这种情况来采取下一步的策略, 这也就是 围棋场上 AlphaGo 能够超越人类的原因

##### 基于概率 和 基于价值

![](https://morvanzhou.github.io/static/results/ML_intro/RLmtd2.png)

基于概率是强化学习中最直接的一种, 他能通过感官分析所处的环境, 直接输出下一步要采取的各种动作的概率, 然后根据概率采取行动, 所以每种动作都有可能被选中, 只是可能性不同. 而基于价值的方法输出则是所有动作的价值, 我们会根据最高价值来选着动作, 相比基于概率的方法, 基于价值的决策部分更为铁定, 毫不留情, 就选价值最高的, 而基于概率的, 即使某个动作的概率最高, 但是还是不一定会选到他.

我们现在说的动作都是一个一个不连续的动作, 而对于选取连续的动作, 基于价值的方法是无能为力的. 我们却能用一个概率分布在连续动作中选取特定动作, 这也是基于概率的方法的优点之一. 那么这两类使用的方法又有哪些呢?

比如在基于概率这边, 有 policy gradients, 在基于价值这边有 q learning, sarsa 等. 而且我们还能结合这两类方法的优势之处, 创造更牛逼的一种方法, 叫做 actor-critic, actor 会基于概率做出动作, 而 critic 会对做出的动作给出动作的价值, 这样就在原有的 policy gradients 上加速了学习过程.

##### 回合更新 和 单步更新

![](https://morvanzhou.github.io/static/results/ML_intro/RLmtd3.png)

强化学习还能用另外一种方式分类, 回合更新和单步更新, 想象强化学习就是在玩游戏, 游戏回合有开始和结束. 回合更新指的是游戏开始后, 我们要等待游戏结束, 然后再总结这一回合中的所有转折点, 再更新我们的行为准则. 而单步更新则是在游戏进行中每一步都在更新, 不用等待游戏的结束, 这样我们就能边玩边学习了.

再来说说方法, Monte-carlo learning 和基础版的 policy gradients 等 都是回合更新制, Qlearning, Sarsa, 升级版的 policy gradients 等都是单步更新制. 因为单步更新更有效率, 所以现在大多方法都是基于单步更新. 比如有的强化学习问题并不属于回合问题.

##### 在线学习 和 离线学习

![](https://morvanzhou.github.io/static/results/ML_intro/RLmtd4.png)

所谓在线学习, 就是指我必须本人在场, 并且一定是本人边玩边学习, 而离线学习是你可以选择自己玩, 也可以选择看着别人玩, 通过看别人玩来学习别人的行为准则, 离线学习 同样是从过往的经验中学习, 但是这些过往的经历没必要是自己的经历, 任何人的经历都能被学习. 或者我也不必要边玩边学习, 我可以白天先存储下来玩耍时的记忆, 然后晚上通过离线学习来学习白天的记忆.那么每种学习的方法又有哪些呢?

最典型的在线学习就是 sarsa 了, 还有一种优化 sarsa 的算法, 叫做 sarsa lambda, 最典型的离线学习就是 Q learning, 后来人也根据离线学习的属性, 开发了更强大的算法, 比如让计算机学会玩电动的 Deep-Q-Network.

## 马尔科夫决策过程

	马尔可夫链/过程，具有markov性质的随机状态序列。
	马尔科夫奖赏过程（MRP），即马尔可夫过程加上value judgement，value judgement是判断一个特定的随机序列有多少累计reward，计算出v(s)，它由[S,P,R,γ]组成。
		
		- S是随机序列中的状态矩阵
		- P是agent的Policy
		- R是随机序列中的累计reward
		- γ是未来时刻reward在当前时刻体现需要乘的discount系数

	v(s)包括两部分，immediate reward和后续状态产生的discounted reward，推导出Bellman方程：

		v(s)=E[Gt|St=s]=E[Rt+1 + γ(Rt+2 + γRt+3 + ...)|St=s]
	
	将Bellman方程表达成矩阵形式，v=ER+γPv,是一个线性等式，直接求解得到v=(I−γP)−1ER,求解的计算复杂度是O(n3)，一般通过动态规划，蒙特卡洛这些迭代的方式求解。

	马尔科夫决策过程（MDP），MRP将所有随机序列都遍历，而MDP是选择性遍历某些情况。

![](http://s8.sinaimg.cn/small/002RSgYjzy7dlTjUYAv17&690)
整个马尔科夫决策过程下来，得到最大的回报价值

![](http://s16.sinaimg.cn/small/002RSgYjzy7dlTjUXSL7f&690)
当选择一个action之后，转移到不同状态下之后获取的reward之和是多少

![](http://s13.sinaimg.cn/small/002RSgYjzy7dlTjUYcA4c&690)
最优状态函数

![](http://s7.sinaimg.cn/small/002RSgYjzy7dlTjUXnU06&690)
最优动作值函数，如果知道了最优动作值函数，也就知道了在每一步选择过程中应该选择什么样的动作。Bellman优化方程，不是一个线性等式，通过值迭代，策略迭代，Q-learning，Sarsa等方式求解。

MDP需要解决的并不是每一步到底会获得多少累计reward，而是找到一个最优的解决方案。上面两个方程是存在这样的关系的，max最优动作函数就能得到最大回报价值。


## 动态规划解决MDP的Planning问题

	MDP的最后问题就是通过Policy iteration和Value iteration来解决。

	Planning是属于Sequential Decision Making问题，不同的是它的environment是已知的，比如游戏，其规则是固定已知的，agent不需要交互来获取下一个状态，只要执行
	action后，优化自己的policy。强化学习要解决的问题就是Planning了。

	动态规划是将一个复杂的问题切分成一系列简单的子问题，一旦解决了子问题，再将这些子问题的解结合起来变成复杂问题的解，同时将它们的解保存起来，如果下一次遇到了相同
	的子问题就不用再重新计算子问题的解。
	其中动态是某个问题是由序列化状态组成，规划是优化子问题，而MDP有Bellman方程能够被递归的切分成子问题，同时它有值函数，保存了每一个子问题的解，因此它能够通过动态
	规划来求解。

	MDP需要解决的问题有两种，一种是policy是已知的，目标是算出在每个状态下的value function，即处于每个状态下能能够获得的reward是多少。
	而第二种是control问题，已知MDP的S，A，P，，R，γ，但是policy未知，目标不仅是计算出最优的value function，而且还要给出最优的policy。

	当已知MDP的状态转移矩阵时，environment的模型就已知了，此时可以看成Planning问题，动态规划是用来解决MDP的Planning问题，主要途径有两种，Policy Iteration
	和Value Iteration。

	-Policy Iteration
	
	解决途径主要分为两步：
		
		- Policy Evaluation：基于当前的Policy计算出每个状态的value function
		- Policy Improvment：基于当前的value function，采用贪心算法来找到当前最优的Policy

	
	-Value Iteration
	
	如果已知子问题的最优值，那么就能得到整个问题的最优值，因此从终点向起点推就能把全部状态最优值推出来。

	针对prediction，目标是在已知的policy下得到收敛的value function，因此针对问题的value iteration就够了。但是如果是control，则需要同时获得最优的policy，
	那么在iterative policy evalution的基础上加如一个选择policy的过程就行了，
	虽然在value itreation在迭代的过程中没有显式计算出policy，但是在得到最优的value function之后就能够推导出最优的policy，因此能够解决control问题。


## Model-Free Learning（解决未知Environment下的Prediction问题）

	已知Environment的MDP问题，也就是已知S，A，P，R，γ，其中根据是否已知policy将问题划分成prediction和control问题，本质上来说这种known MDP问题已知
	Environment即转移矩阵与reward函数，还是很多问题中Environment是未知的，
	不清楚作出某个action之后会变成什么state，也不知道这个action好还是不好，也就说不清楚Environment体现的model是什么，在这种情况下需要解决的
	prediction和control问题就是Model-Free prediction和Model-Free control。显然这种新的问题只能从与Environment的交互得到experience中获取信息。

	-Model-Free prediction
	
	未知Environment的policy evalution，在给定的policy下，每个state的value function是多少。

	episode：将从某个起始状态开始执行到终止状态的一次遍历S1，A1，R2，...，Sk成为episode。

	-Monte-carlo Reinforcement Learning
	
	蒙特卡洛强化学习是假设每个state的value function取值等于多个episode的return Gt的平均值，它需要每个episode是完整的流程，即一定要执行到终止状态。

	v(s)=E[Gt|St=s]
	
	回报价值函数在蒙特卡洛的假设下，值函数的取值从期望简化成了均值。

	-Temporal-Difference Learning
	时序差分学习则是基于Bootstarpping思想，即在中间状态中会估计当前可能获得的return，并且更新之前状态能获得的return，因此不需要走完一个episode的
	全部流程才能获得return。

	-Bias/Variance tradeoff
	bias指预测结果和真实结果的差值，variance指训练集每次预测结果之间的差值，bias过大会导致欠拟合它衡量了模型是否准确，variance过大会导致过拟合衡量了
	模型是否稳定。
	如果回报价值函数值Gt跟真实值一样，那么就是无偏差估计，因此在蒙特卡洛算法中，将最终获得的reward返回到了前面的状态，因此是真实值，但是它采样的episode
	并不能代表所有的情况，所以会导致比较大的variance。
	而时序差分算法的回报价值函数值跟真实值是有偏差的，在计算的过程基于随机的状态，转移概率，reward等，涵盖了一些随机的采样，因此variance比较小。
	蒙特卡洛方法中没有计算状态转移概率，也不考虑状态转移，它的目标是最小化均方误差，这样的行为实际上并不符合马尔科夫性质，而时序差分方法会找出拟合数据
	转移概率和reward函数，还是在解决MDP问题。

![](http://7xkmdr.com1.z0.glb.clouddn.com/rl4_4.png)


## Model-Free Control（解决未知Environment下的Control问题）

	解决未知policy情况下未知Environment的MDP问题，也就是Model-Free Control问题，这是最常见的强化学习问题。
	
	-On-Policy Monte-Carlo
	
	动态规划解决planning问题（已知Environment）中提出policy iteration和value iteration，其中policy itertion和policy improvenment组成。
	
	未知Environment的policy evaluation是通过蒙特卡洛方法求解，结合起来得到一个解决Model-Free control方法，先通过贪婪算法来确定当前的policy，
	再通过蒙特卡洛policy evaluation来评估当前的policy好不好，再更新policy。

	在已知Environment情况下，policy improvement更新的解决办法是通过状态转移矩阵把所有可能转移到的状态得到的价值函数值都计算出来，选出最大的。
	但未知environment没有状态转移矩阵，因此只能通过最大化动作价值函数来更新policy，由于improvement的过程需要动作值函数，那么在policy evaluation
	的过程中针对给定的policy需要计算的回报价值函数V(s)也替换成动作值函数Q(s,a)。
		
![](http://s11.sinaimg.cn/small/002RSgYjzy7dm7BYFtg4a&690)
	
	-Sarsa Algorithm
		
	具体的On-Policy Control流程如下：

		- 初始化Q(s,a)
		- for each episode:
		- ==初始化一个状态S
		- ==基于某个策略Q和当前状态S选择一个动作A
		- == for each step of one episode:
		- ====执行一个动作A，得到反馈的immediate reward为R，和新的装填S'
		- ====基于当前策略Q和状体S'选择一个新动作A'
		- ====更新策略：
![](http://s11.sinaimg.cn/small/002RSgYjzy7dm7BYFtg4a&690)
		
		- ====更新状态S=S'
		- ==直到S到达终止状态

	-Off-Policy Learning

	Off-Policy Learning是在某个已知策略（behaviour policy）μ(a|s) 下来学习目标策略(target policy)π(a|s),这样就可以从人的控制或者其他表现的
	比较好的agent中来学习新的策略。

	已知策略分布P(X),目标策略分布Q(X),reward函数f(X),两种分布中reward期望为Exp[f(X)],从μ中来估计π获得的return，此方法称为Importance Sampling。

	同样的Off-Policy TD也是改变了更新值函数公式，改变的这一项相当于给TD target加权，这个权重代表了目标策略和已知策略匹配程度，代表了是否能够信任目标
	policy提出的这个action。

![](http://s7.sinaimg.cn/small/002RSgYjzy7dm8KD1I256&690)
		
	-Off-Policy Q-Learning
	针对未知policy，Off-Policy的解决方案是Q-Learning，更新动作值函数。在某个已知策略下选择了下一个时刻的动作At+1，以及下一个时刻的状态St+1和奖赏Rt+1,
	将目标策略选择的动作A'替换到更新公式中。
	与上面方法不同的是，可以同时更新π和μ，且π是greedy的方式，而μ是采用了ε-greedy方式。

	
	
## DQN

	现实中，不少问题的状态S的取值和动作A非常多，例如围棋的361个定位，每个定位会出现黑白空三种情况，如果计算每种状态下的真实value function既没有足够的
	内存也没有足够的计算能力，需要算法来求解近似的V(S)和Q(S,A)，并且针对未知的状态有比较强的泛化能力。
	这种近似算法称之为function approximation，神经网络为例，即把近似值函数用神经网络实现出来。

Deep Q-Networks，具体算法如下：


	- 根据ε-greedy policy选择一个动作at（根据Q-Learning，这里应该是behaviour policy）
	- 选择完at后产生下个时刻的状态和奖赏，将多个转义序列保存在称为reply memory的集合D中
	- 从D中随机选择一些转移序列，基于这些和固定参数ω计算Q-Learning的target
	- 通过随即梯度下降方法来优化Q-Learning的target和近似函数的均方差。

	相比于传统的强化学习方法，DQN融合了神经网络和Q learning的方法，传统的表格形式的强化学习有一个瓶颈。
	
	我们使用表格来存储每一个状态state，和这个state每个行为action所拥有的Q值，而问题复杂时，状态就比天上的星星还多，
	比如下围棋，内存不足以存储所有状态，此外在这么大的表格中搜索对应的状态也是一件很耗时的事。神经网络可以将状态和动作当成神经网络的输入，然后经过神经网络
	分析后得到的Q值，还有一种形式是这样的，我们只输入状态值，输出所有的动作值，然后按照Q learning的原则，直接选择拥有最大值的动作当作下一步要做的动作。

![](https://morvanzhou.github.io/static/results/ML_intro/DQN3.png)

Q现实用之前在Q larning中的Q现实来代替。还需要一个Q估计来实现神经网络的更新，神经网络的参数就是老的NN参数加学习率alpha乘以Q现实和Q估计的差距

![](https://morvanzhou.github.io/static/results/ML_intro/DQN4.png)

通过NN预测出Q(s2,a1)和Q(s2,a2)的值，这就是Q估计，然后选取Q估计中最大值的动作换取环境中的奖励reward。DQN还有两个因素使其无比强大：Experience replay和Fixed Q-targets。

	DQN有一个记忆库用于学习之前的经历，Q learning是一种off-policy离线学习方法，所以每次DQN更新的时候，我们都可以随机抽取一些之前的经历进行学习，随机抽取
	这种做法打乱了经历之间的相关性，也使得神经网络更有效率，
	Fiexed Q-targets也是一种打乱相关性的机理，如果使用fixed Q-targets，在DQN中使用两个结构相同但参数不通的神经网络，预测Q估计的神经网络具备最新的参数，
	而预测Q现实的神经网络使用的参数则是很久以前的。


## Policy Gradient

	将policy看成某个参数θ的函数，即将policy形式变成状态和动作的概率分布函数，在policy函数可微的情况下能够通过对参数求导来优化policy。

	将value function，policy，model（environment）进行组合可以得到model-based，policy-based，model-free，value-based，actor critic五种类型，
	其中value-based是说已知policy情况下学习value function。policy-based是指没有显式的值函数形式，需要学习policy。
	actor critic则是需要同时学习值函数和policy的形式。

	policy都是基于greedy或者ε-greedy的方法直接从值函数中获得。
	现实中有大量的state和action，无法针对每个state每个action都有一个确定的policy，因此需要一定的泛化能力，面对没有见过的state或者action有一定的决策能力。
	为policy引入采纳数，变成在某个状态和某组参数下选择某个动作的概率分布，直接求解策略，学习如何让 policy变得更好。

	衡量policy好坏有三种方法，一是在某个状态下和该policy作用下能获得的值函数值，一是该policy作用下能获得的所有状态的期望值函数，一是在该policy作用下
	能获得的所有状态的期望immediate reward，推导出这三种方法的统一导数形式，这就是policy gradient。

	Policy gradient是RL中另外一个大家族，他不像value-bsed方法（Q-Learning，Sarsa），他也要接受环境（observation），不同的是他要输出的不是action的value，而是具体的那一个action，这样policy gradient就跳过了value这个阶段。
	Policy gradient最大的一个优势就是：输出的action可以是一个连续的值，value-based方法输出的都是不连续的值，然后再选择值最大的action。
	而policy gradient可以在一个连续分布上选取action。

##### Monte-Carlo Policy Gradient

	目标是优化reward，也就是优化值函数，只是这里θ不是值函数的参数，而是policy的参数，如果目标函数对参数求导，得到policy的gradient的形式为：

![](http://s4.sinaimg.cn/small/002RSgYjzy7dpknZ1R1a3&690)	
	
	第一项是衡量policy朝当前选择（某个状态+某个动作）偏移的程度，第二项衡量了当前选择的好坏。

	从而推导出Monte-Carlo Policy Gradient的形式，首先更新参数的方法是随机梯度下降+policy gradient，gradient中的动作值函数取值用执行过程中的return来代替。

##### Actor-Critic Policy Gradient

	Monte-Carlo Policy Gradient是用episode中反馈的return当作是动作值函数的采样，如果采用value fcuntion approximation的方法，即迭代更新policy又更新值函数。


## Integrating Learning and Planning（对Environment建立模型）

	如何拟合environment模型，通过有监督的方式来更新model，以及如何基于学习的model来找policy/value function，谈到Monte-Carlo Tree Search方法，
	将拟合model和求解value function结合起来实现Dyna算法。

	model-based Reinforcement learning：之前都是将policy和value function表达成参数形式，从而通过更新参数来解决MDP问题，
	而本科希望拟合出未知environment的model，即知道environment是如何转移状态和反馈reward，就能从model中找到合适的policy和value function，这步称为planning。
	plan就是学习model，并从model中推断出policy和value function。

	environment的参数化好处在于，避免了状态多policy复杂难学，它像在模拟出游戏的规则，这样就能够通过搜索树的方式获得value function，并且能够通过监督学习方式来学习model。
	例如输入是状态s输出是状态s'，此处就能给一定的监督信息，说明s'取什么样的值比较好，而缺点在于用一个近似的model来构造一个近似的value function，两个近似都会存在一定的误差。

#### Model-Based Reinforcement Learning

	model的形式可以表现为S,A,P,R，假设状态集和动作集是已知的，把状态转移和奖赏表示成状态和动作的独立概率分布。目标就是从一段执行序列中S1,A1,R2,...，St来估计model，
	这样就看成一个监督问题，把St，At当成模型输入，那么即Rt+1，St+1就能看成模型的输出。
	学习Rt+1是一个回归问题，可以用最小均方差等方法来解决，而学习St+1是一个密度估计问题，可以用KL divergence等方法来解决。

#### Integrated Architectures

	数据存在两种版本，一种是真实数据称为Real experience，一种是从model中采样出来的称为Simulated experience。

	如果将learn和plan结合起来，即又学习model的形式，又学习value function或者policy，这种方法称为Dyna。
	experience包括real和simulated两种，既可以用real数据来直接学value function/policy，也可以将real数据用来学习modle，然后由model产生simulated数据继续学
	value function/policy。
	结合Q-learning的算法流程如下：

![](http://7xkmdr.com1.z0.glb.clouddn.com/rl_8_2.png)

#### Simulation-Based Search

	上图在plan值函数的时候，是随机选状态和动作再更新。但是在状态集合常常会有比较关键的状态和没用的状态，那么就应该更多的关注重要的状态，将一个重要的状态作为根节点，
	由Model(s,a)得到从这个状态出发的搜索树，再从分支中选择一些episode来做model-free learning，
	如果选择的方法是Monte-Carlo control方法，那么整个算法就叫Monte-Carlo Search。实际上就是将Monte-Carlo control应用到simulated experience上。
	如果MC的部分改成TD来更新Q(s,a)就是TD search。
	Dyna算法就是分两次更新值函数。


## Exploration and Exploitation

	Exploration会尝试新的选择，而Exploitation会从已有的信息中选择最好的。比如买股票，前者会尝试一些新股，而或者会选择收益最高的股。

	有三种方式达到exploration的目的，第一种就是类似ε-greedy算法在一定概率基础下随机选择一些action，第二种是更加倾向选择更加具有不确定性的状态/动作，这种方法
	需要一种方法来衡量这种不确定性，第三种就是收集一些状态信息来确定是否值得到达这个状态。

#### εt-greedy
	
	实验过程中t时刻动作值函数是每次reward的均值，与最好的取值的差的期望称为regret。最大化累计奖赏就是最小化total regret，regret就能衡量算法能够好到什么程度。
	
	ε-greedy算法就是避免总是选已知信息中最好的，不要让regret出现线性，但是还是避免不了regret呈线性，将ε的值不固定，而是采用逐渐减少的εt，会让regret呈现对数函数形式。

#### Upper Confidence Bounds Algorithm

	第二种方法，选择不确定性的action可以，但是统计这些被选中的次数count的大小来衡量，如果count比较大，说明这个action能多次被利用，带来比较好的reward。

#### Information State Search

	根据第三种方法，考虑当前获取的information。




	

	