# <center>Enforcement learning</center>

## 强化学习的术语和数学符号
#### 符号
- agent  学习者，决策者
- enviroment  环境
- s/state  状态，表示环境的数据
- S  所有状态的集合
- a/action  决策者的动作
- A  所有行动的集合
- A(s) 状态s的行动集合
- r/reward agent在一个action之后，获得的奖赏
- R  所有奖赏集合
- St 第t步的状态
- At 第t步的行动
- Rt 第t步的奖赏
- Gt 第t步的长期回报，也是强化学习的目标1，最求最大长期回报


- π 策略policy，策略规定了状态s时，应该选择动作a，强化学习的目标2，找到最优策略
- π(s) 策略π在状态s下，选择的行动
- π* 最优策略optimal policy
- r(s,a) 在状态s下，选择行动a的奖赏
- r(s,a,s') 在状态s下，选择行动a，变成状态s'的奖赏
- p(s'|s,a) 在状态s，选择行动a的前提下，变成状态s'的概率
- vπ(s) 状态价值，使用策略π，状态s下的长期奖赏Gt
- qπ(s,a) 行动价值，使用策略π，状态s，选择行动a下的长期奖赏Gt
- v*(s)  最佳状态价值
- q*(s,a) 最佳行动价值  强化学习的目标3：找到最优价值函数或者最佳行动价值函数
- V(s) vπ(s)的集合
- Q(s,a) qπ(s,a)的集合

- v^(St,θt)  最优近似状态价值函数
- q^(St,At,θt)  最优近似行动价值函数  强化学习的目标4：找到最优近似状态价值函数或者最优近似行动价值函数

- θ 近似价值函数的权重向量   强化学习的目标5：找到求解θ
- φ(s) 近似状态价值函数的特征函数，是一个将状态s转化成计算向量的方法，其和θ组成近似状态价值函数

		v^ ≈ transpose(θ)φ(s)

-  φ(s,a) 近似行动价值函数的特征函数，是一个将状态s，行动a转化成计算向量的方法，其和θ组成近似行动价值函数

		v^ ≈ transpose(θ)φ(s,a)

- et  第t步的有效跟踪向量(eligibility trace rate),可理解为近似价值函数微分的优化值。

		e0 ≈ 0
		et ≈ ▽v^(St,θt) + γλet-1
		θt ≈ θt + αδtet

- α  学习步长 α∈(0,1]
- γ 未来回报的折扣率（discount rate）γ∈[0,1]
- λ λ-return 中的比例参数 λ∈[0,1]
- h horizon,水平线h表示on-line当时可以模拟的数据步骤。 t<h≤T
- ε 在ε-greedy策略中，采用随机行动的概率 ε∈[0,1)

#### 术语
- episodic tasks  情节性任务，指会在有限步骤下结束
- continuing tasks 连续性任务，指有无限步骤
- episode 情节，指从起始状态（或者当前状态）到结束的所有步骤
- tabular method 列表方法，指使用了数组或者表格存储每个状态（或者状态-行动）的信息（比如：其价值）
- approximation methods 近似方法，指用一个函数来计算状态（或者状态-行动）的价值
- model  环境的模型，可以模拟环境，模拟行动的结果 Dynamic Programming need a model
- model-based methods 基于模型的方法，通过模型来模拟，可以模拟行动，获得状态或者行动的价值
- model-free methods 无模型的方法，使用试错法（trial-and-error）来获得（状态或者行动）价值
- bootstarpping 引导性 （状态或者行动）价值是根据其他的（状态或者行动）价值计算得到的
- sampling 取样性  （状态或者行动）价值，或者部分值（比如：奖赏）是取样得到的。引导性和取样性并不是对立的，可以是取样的，并且是引导性的
- planning method 计划性方法，需要一个模型，在模型里，可以获得状态价值，比如：动态规划
- learning method 学习性方法，不需要模型，通过模拟（或者体验），来计算状态价值，比如：蒙特卡洛方法，时序差分方法
- on-policy method  on-policy方法，评估的策略和优化的策略是同一个
- off-policy method off-policy方法，评估的策略和优化的策略不是同一个，意味着优化策略使用来自外部的样本数据
- predication algorithms 预测算法，计算每个状态的价值v(s),然后预测能得到最大回报的最优行动。
- control algorithms 控制算法，计算每个状态下每个行动的价值q(s,a)
- target policy 目标策略π，off-policy方法中需要优化的策略
- behavior policy 行为策略μ，off-policy方法中提供样本数据的策略
- importance sampling 行为策略μ的样本数据
- importance sampling rate 由于目标策略π和行为策略μ不同，导致样本数据在使用上的加权值
- ordinary importance sampling 无偏见的计算策略价值的方法
- weighted importance sampling 有偏见的计算策略价值的方法
- MSE(mean square error) 平均平法误差
- MDP(markov decision process) 马尔科夫决策过程
- the forward view 通过往前看，直到将来，根据其回报和状态来更新每一步的状态，
- the backward or mechanistic view  根据current TD error集合上过往的有效跟踪(eligibility traces)来更新当下的有效跟踪
		
		e0 ≈ 0
		et ≈ ▽v^(St,θt) + γλet-1

## 强化学习分类和基本组成

- 如果有一个模型，可以获得价值函数v(s)或者q(s,a)的值---> 动态规划方法
- 如果可以模拟一个完整的episode ---> 蒙特卡洛方法
- 如果需要在模拟一个episode中间就要学习策略 ---> 时序差分方法
- λ-retrun用来优化近似方法中的误差
- 有效跟踪(eligibility traces)用来优化近似方法中的价值函数的微分
- 预测方法是求状态价值方法v(s)或者v^(s,θ)
- 控制方法是求行动价值方法q(s,a)或者q^(s,a,θ)
- 策略梯度方法(policy gradient methods)是求策略方法π(a|s,θ)

	在第t步agent的工作流程是执行一个动作at，获得该动作之后的环境观测状况st，以及获得这个动作的反馈奖赏rt。然后不断累积反馈奖赏且最大化。

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

## Multi-arm Bandits

### 什么是多臂老虎机？

多臂老虎机是指一类问题，该类问题重复的从k个行为action中选择一个，并获得一个奖励reward，一次选择的时间周期是一个时间步长（time-step）。当选择并执行完一个action后，得到一个reward，我们称reward的期望为这个action的真实值value。

	q*(a) = E[Rt | At=a]

如果知道每个行为的真实值，那么多臂老虎机的问题就很容易解决，但是大多数情况下，我们是不知道行为的具体值的，因此只能做近似：q^(a) ≈ q*(a)

在时刻t，利用已有的知识（即行为的估计值）进行action的最优选择，这种操作称为exploit，如果不选择当前的最优行为，我们称之为explore，explore操作能够提高对行为值估计的准确度。exploit操作能够最大化当前步的奖励，但explore操作可能会使得长期的奖励更大。也就是说explore一下，说不定会有大的惊喜。如果平衡exploit操作和explore操作是强化学习中的一个重要问题（exploitation-exploration dilemma）。

### 估计行为值的方法

对行为值的估计是为了更好的选择行为，行为的值为每次执行该行为所得奖励的期望，因此可以用t时刻前行为已得到的奖励作为行为值的估计

	q^t(a) = t时刻前a的reward之和 / t时刻前a的出现总数

上面这种方法是**样本平均（sample-average）法**，在t时刻选择行为时，使用贪心策略来选择行为值最大的行为，即：

	At = argamaxq^t(a)

**greedy法**有一个缺陷，制作exploit操作，而不做explore操作，选择行为时可能会漏掉那些真实值更大的行为（惊喜）。改进就是**ϵ-greedy方法**

如果行为的奖励恒定不变的话，样本平均法就能解决问题，但这种情况显然不多，大多数情况下行为的奖励是服从某个分布的，甚至是非平稳（nonstationary）问题，其中行为的真实值会发生变化，显然在这种情况下ϵ-greedy方法比样本均值法能获得更好的效果。

采用增量式的行为估计值：

![](http://s16.sinaimg.cn/middle/002RSgYjzy7erkMu2yXcf&690)

Ri是第i次执行该行为后所得到的奖励。并不会浪费空间和时间来每次都直接计算Qn.

上面公式的更新方式可以总结如下：

	新估计值 <- 旧估计值 + 步长 * (目标值 - 旧估计值)

其中（目标值 - 旧估计值）为误差（error）

### 初始值

对于平稳问题，前面两个方法在使用中有一个小技巧，就是提高初始值的大小。通过提高初始的行为估计值，可以有效的促进explore操作。比如，假设多臂赌博机的行为值服从期望为0，方差为1的正态分布，那么我们可以将初始的行为估计值设为5，当系统选择一个行为开始执行后，所获得的奖励很可能比5小，因此接下来就会尝试其他估计值为5的行为。这样在估计值收敛前，所有的行为都已经被执行了一遍或多遍。 

通过设置较高的初始值是一种解决平稳问题的有效方法，但对于非平稳问题就没有那么好的效果。因为非平稳问题的真实行为值会发生变化

### 置信上界行为选择

ϵ-greedy方法能够迫使agent执行explore操作，但存在一个问题，即进行explore操作时，如何选择那些不具有最高估计值的行为。一种思路是同时考虑行为的估计值与最大行为估计值的差距以及估计过程中的不确定性。用公式表示为： 

![](http://s15.sinaimg.cn/middle/002RSgYjzy7erkMu1Ii0e&690)

其中Nt(a) 表示在t 时刻前行为a 被执行的次数，Nt(a) 越大说明行为a 的估计值被更新的次数越多。如果Nt(a)=0 则认为行为a 首先被执行。 
上述在explore操作中选择行为的思路称为**置信上界（upper confidence bound, UCB）**，根号中的部分表示估计过程中的不确定性，因为估计的次数越少就代表不确定性越高，之所以采用t 的自然对数是为了减小根号内部分的增长速度。这种思路会使所有的行为都被执行过。UCB方法在实际使用的过程中也取得了很好的效果，但相比与ϵ−greedy方法扩展性较差，而且不易应用到非平稳问题中。

## 马尔科夫决策过程

马尔可夫链/过程，具有markov性质的随机状态序列。
	
	The Markov Property:理想情况下，一个state信号，很好地总结了过去的感知，所有相关信息都被保留下来了，这通常比即时的感知更多，但不会超过所有历史感知的总和。
	一个成功保留所有相关信息的state信号，称它为Markov，具有Markov property。

MDP：满足Markov property的增强学习任务我们称为Markov Decision Process。如果state和action的空间是有限的，那么称为finite Markov decision process（finite MDP）。

一个特定的finite MDP由state集合S和action集合A以及enviromen的one-step dynamics（一步变换）p(s',r|s,a)定义，即在给定state和action s和a下，下一个state和reward s',r的概率。

马尔科夫奖赏过程（MRP），即马尔可夫过程加上value judgement，value judgement是判断一个特定的随机序列有多少累计reward，计算出v(s)，它由[S,P,R,γ]组成。
		
- S是随机序列中的状态矩阵
- P是agent的Policy
- R是随机序列中的累计reward
- γ是未来时刻reward在当前时刻体现需要乘的discount系数

v(s)包括两部分，immediate reward和后续状态产生的discounted reward，推导出Bellman方程：

		v(s)=E[Gt|St=s]=E[Rt+1 + γ(Rt+2 + γRt+3 + ...)|St=s]
	
将Bellman方程表达成矩阵形式，v=ER+γPv,是一个线性等式，直接求解得到v=(I−γP)−1ER,求解的计算复杂度是O(n3)，一般通过动态规划，蒙特卡洛这些迭代的方式求解。

马尔科夫决策过程（MDP），MRP将所有随机序列都遍历，而MDP是选择性遍历某些情况。

![](http://s8.sinaimg.cn/middle/002RSgYjzy7dlTjUYAv17&690)

上式为Bellman equation for vπ，刻画了state s与它可能转移到的下一个state s'之间，value的关系。表示了state和value与它后继state的value之间的关系。从s开始，agent可以依照policy π选择action，对于每个state-action，enviroment可以依概率p(s'|s,a)转移到不同的s'，给出相应的reward r。Bellman equation实际上就是对所有转移可能求一个加权平均，权值即为每一个选择的概率。表明：**当前state的value等于下一个state的期望value +  转移过程中reward的期望。**

整个马尔科夫决策过程下来，得到最大的回报价值

![](http://s16.sinaimg.cn/middle/002RSgYjzy7dlTjUXSL7f&690)

当选择一个action之后，转移到不同状态下之后获取的reward之和是多少

![](http://s13.sinaimg.cn/middle/002RSgYjzy7dlTjUYcA4c&690)

最优状态函数

![](http://s7.sinaimg.cn/middle/002RSgYjzy7dlTjUXnU06&690)

vπ和qπ可以由经验估计，比如一个agent遵循policy π，对于每个遇到的state，它维护这个state之后获得return的均值，随着遇到这个state的次数趋向于无穷，均值将收敛到这个state实际的value。

如果将每个state的均值根据选择的action不同分成很多种，那么每一种就对应了这个state下采取特定的aciton和value，qπ(s,a)。我们称这种方法为**Monte Carlo methods**，因为它涉及到对实际的return做多次随即采样后取均值。

当然，当state集合很大的时候，这种方法就不切实际了，或许我们需要对state进行参数化，获得一个更小的参数集合替代state集合，参数下的vπ，qπ估计return的精确度取决于参数化函数的近似性质。

最优动作值函数，如果知道了最优动作值函数，也就知道了在每一步选择过程中应该选择什么样的动作。Bellman优化方程，不是一个线性等式，通过值迭代，策略迭代，Q-learning，Sarsa等方式求解。

MDP需要解决的并不是每一步到底会获得多少累计reward，而是找到一个最优的解决方案。上面两个方程是存在这样的关系的，max最优动作函数就能得到最大回报价值。

解决一个增强学习任务，简单地讲就是找到一个具有长远回报的策略，对于finite MDP，可以精确地定义一个optimal policy（最优策略）。

上面的最优状态函数v*,q*，不同于vπ，qπ，我们用max操作取代了加权平均。

对于finite MDP，最优价值函数v*和最优状态函数q*，Bellman optimality equation实际上是一个方程组，对于N个states，那就有N个方程，N个未知量，因此如果能够直到enviroment的变化机制p(s',r|s,a)，原则上，我们就可以用任何一种解非线性方程组的方法得到v*,q*。

得到v*，确定optimal policy是非常容易的，对于每个state，将一个或多个action，满足通过Bellman optimality equation解得的最大值，任何只对这些aciton赋予非零π(a|s)的policy都是一个optimal policy。

对于v*，使用greedy算法即可得到optimal policy。所谓的greedy，即基于局部或即时的信息作出选择，而不考虑这个选择是否放弃了那些长远来看更好的选择。v*的优雅之处在于，根据它得到的局部最优解，到最后就是全局最优解。

![](http://s8.sinaimg.cn/middle/002RSgYjzy7evEhwFgz77&690)

上面为Bellman Optimality Equation，刻画了state s与它可能转移到的下一个state s'之间，optimal value的关系。在贝尔曼最优方程中，v*(s)已经跟policy无关。只与由enviroment决定的p(s',r|s,a)有关，因此，如果已经知道p(s',r|s,a)，就可以得到N个state和N个方程，就可以通过解非线性方程组得到v*，从而求得π*。


## 动态规划（Dynamic Programming）

动态规划是计算最优策略的一组算法

所谓最优策略就是取得最大的长期奖赏的策略

对策略进行价值计算，有两种计算方式，一种是策略的状态价值计算，一种是策略的行动价值计算，前者有利于发现哪个状态价值高，找到最优状态。后者有利于发现特定状态下哪个行动的价值高，找到最优行动。

### 通用策略迭代（Generalized Policy Iteration）

- 先从一个策略π0开始
- 策略评估（policy evaluation）得到策略π0的价值V0
- 策略改善（policy improvement）根据V0，优化策略π0
- 迭代上面的2，3步，直到找到最优价值V*，因此得到最优策略π*

通用策略迭代（GPI）的通用思想是：两个循环交互的过程，迭代价值方法（value function）和迭代优化策略方法。

动态规划（DP）对复杂的问题来说，可能不具有可行性，主要原因是问题状态的数量很大，导致计算代价太大。


## 动态规划解决MDP的Planning问题

DP是一个算法的集合，一个强假设是有一个完美的符合马尔科夫性质的环境模型，用来计算最优策略π。经典的DP既要强假设一个完美环境模型，同时其计算昂贵，但作为理论基础是极其重要的，后续的算法，本质上都是实现DP的同样功能，只是去掉这个完美环境模型的强假设或者减少计算代价而已。

DP，通常假设环境是finite MDP。也就是说state，action和reward set是finite，且其动态转化是给定了一个概率集合p(s',r|s,a)。当然DP想法也能够解决连续state和连续action问题。

The key idea of DP,and of reinforcement learning generally,is the use of value functions to organize and structure the search for good policies.

##### 什么是full backup？
为了从vk产生每一个延续的近似值vk+1，迭代策略等式对每一个状态s应用相同的操作：用一个新的value替换老的value，而这个新的value是从老的状态s，预期的当下reward，在已知策略policy经过状态转移one-step一步一步估计得到的。之所以叫full backup，就是因为其构建在所有可能的下一个state基础上，而不仅仅只是下一个state。

MDP的最后问题就是通过Policy iteration和Value iteration来解决。

Planning是属于Sequential Decision Making问题，不同的是它的environment是已知的，比如游戏，其规则是固定已知的，agent不需要交互来获取下一个状态，只要执行action后，优化自己的policy。强化学习要解决的问题就是Planning了。

动态规划是将一个复杂的问题切分成一系列简单的子问题，一旦解决了子问题，再将这些子问题的解结合起来变成复杂问题的解，同时将它们的解保存起来，如果下一次遇到了相同的子问题就不用再重新计算子问题的解。

其中动态是某个问题是由序列化状态组成，规划是优化子问题，而MDP有Bellman方程能够被递归的切分成子问题，同时它有值函数，保存了每一个子问题的解，因此它能够通过动态规划来求解。

	
MDP需要解决的问题有两种：

- 一种是prediction问题，policy是已知的，目标是算出在每个状态下的value function，即处于每个状态下能能够获得的reward是多少。
- 而第二种是control问题，已知MDP的S，A，P，，R，γ，但是policy未知，目标不仅是计算出最优的value function，而且还要给出最优的policy。

当已知MDP的状态转移矩阵时，environment的模型就已知了，此时可以看成Planning问题，动态规划是用来解决MDP的Planning问题，主要途径有两种：Policy Iteration和Value Iteration。

- Policy Iteration
 
	π0 --E--> vπ0 --I--> π1 --E--> vπ1 --I--> π2 --E--> ... --I--> π* --E--> v*
 	--E--> 表示policy evaluation
	--I--> 表示a policy improvement

![](http://s7.sinaimg.cn/middle/002RSgYjzy7evT7Cp94b6&690)

 解决途径主要分为两步：
		
	- Policy Evaluation：基于当前的Policy计算出每个状态的value function
	- Policy Improvment：基于当前的value function，采用贪心算法来找到当前最优的Policy

	上面两步不断迭代构成一个序列，这个序列一定会在有限步数的迭代之后收敛到一个optimal policy，并得到一个optimal value function。这种寻找optimal policy的方法称之为policy iteration。
	
	Policy Iteration的一个弊端是它需要每一轮迭代都涉及到policy evaluation，而这本身又需要扫遍整个状态集的耗时迭代计算过程。
	
- Value Iteration

![](http://s9.sinaimg.cn/middle/002RSgYjzy7evUffbDO08&690)
	
如果已知子问题的最优值，那么就能得到整个问题的最优值，因此从终点向起点推就能把全部状态最优值推出来。

针对prediction，目标是在已知的policy下得到收敛的value function，因此针对问题的value iteration就够了。但是如果是control，则需要同时获得最优的policy，那么在iterative policy evalution的基础上加如一个选择policy的过程就行了，虽然在value itreation在迭代的过程中没有显式计算出policy，但是在得到最优的value function之后就能够推导出最优的policy，因此能够解决control问题。

DP的不足之处，就是在每一轮迭代时，会涉及到整个状态集，如果状态集非常大，那么即便是一轮简单的迭代，也会有高昂的开销。

##### Asynchronous Dynamic Programming

	vk+1(s) ≈ maxE[Rt+1 + γvk(St+1) | St=s,At=a]

Asynchronous DP不完整地遍历每个state作为一轮迭代，它每一次任意地选择一个state然后根据上面Value Iteration的主要公式更新它的value function。也就是说Asynchronous DP更灵活，它可以按任意次序迭代地更新state。Asynchronous DP的优势在于，它可以加速收敛的速度，因为它可以有意识地挑选收敛更快的state进行更新，另外，它也很适合于实时交互的场景，它可以在有限时间内给出一个较优的policy。

##### Generalized Policy Iteration

Policy Iteration由两步构成，一步是通过当前policy计算相应的value function，一步是根据当前的value function改进policy。在policy iteration中，这两个步骤交替进行，必须在对方完成后开始进行。在value iteration中，我们看到两个步骤交替进行是没有必要的，policy evaluation可以在改进policy的同时进行。而Asynchronous DP方法更是将policy evaluation和policy improvement的交替执行划分到更细的粒度。无论如何，最终的结果都是收敛到optimal value function和optimal policy。

generalized policy iteration（GPI）来描述让policy evaluation和policy improvement相互交互的想法，与具体的粒度无关。

![](http://img.blog.csdn.net/20170329153401442?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHViaW4wMHN4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在GPI中，evaluation和improvement之间的关系可以看作是竞争和合作的关系，它们朝两个相反的方向相互竞争，根据value function改进的policy会使得在新policy下原来的value function不再是最优的，而根据policy计算得到的value function会使得当前的policy不再是最优的，他们是相互竞争的，但它们最终会收敛到到同一点，因此也是相互合作的。

##### Summary

对于大规模的问题，DP并不实用，但是相比较其他解MDP的方法，DP实际上还是相当高效的。如果我们不考虑一些技术上的细节，DP寻找以恶搞optimal policy的最坏时间，是关于state和action的数量的多项式。假设n和k是states和actions的数量，尽管可能的policies的数量是指数k**n的，但DP的时间复杂度是n和k的多项式时间。

线性规划方法（linear Programming methods）也可以用来解MDP，而且在一些实例上，他们的时间复杂度要优于DP，但是线性规划的方法在states数量增加的时候变得不切实际，因此在较大规模的问题上DP更合适。

但DP也会有不适用的时候，主要是因为维度灾难（curse of dimensionality）。实际上，states的个数与state变量的个数是成指数关系的，state变量相当于维数，每扩展一维，states的数量就呈指数增长。然而，DP比直接暴力搜索和线性规划方法更适合处理state空间大的问题。

在实际应用中，使用现在的计算机，DP可以解百万数量级states的MDP问题，在较大state空间的问题下，我们更加倾向使用Asynchronous DP，对于synchronous DP，一轮计算都需要在每一个state上消耗计算和存储，对于有些问题，可能即使是一轮的空间和时间开销都是无法满足的。但这些问题依然有可解的可能性，因为实际上只有相对较少的states会出现在optimal solution的轨迹上。synchronous DP或者GPI的一些其他变体可以在这些问题上应用，会比synchronous方法更快地得到一个较优或最优policy。

## 蒙特卡洛

蒙特卡洛方法的整体思想就是：模拟 -> 抽样 -> 估值

不像之前内容，MC不再假设对环境的完全了解。蒙特卡洛方法只需要经验，就是对状态states，动作actions和奖赏rewards进行采样，不管其是来自实际或者别的交互经验都是可以的。

The term "Monte Carlo" is often used more broadly for any estimation method whose operation involves a significant random component.

Monte Carlo methods sample and average returns for each state-action pair much like the bandit methods that sample and average rewards for each action.
The main difference is that now there are multiple states,each acting like a different bandit problem and that the different bandit problems are interrelated.

下面是强化学习中蒙特卡洛方法的一个迭代过程：

	- 策略评估迭代
		- 探索，选择一个状态和一个动作（s,a）
		- 模拟，使用当前策略π，进行一次模拟，从当前状态(s,a)到结束，随机产生一段情节（episode）
		- 抽样，获得这段情节上的每个状态(s,a)的回报R(s,a),记录R(s,a)到集合Returns(s,a)
		- 抽样，q(s,a) = Returns(s,a)的平均值
	- 策略优化,使用新的行动价值q(s,a)优化策略π(s)


上面的蒙特卡洛方法，其中模拟过程是会模拟到结束，需要大量的迭代，才能正确找到最优策略。

exploring start假设，有一个探索起点的环境。比如围棋的当前状态就是一个探索起点，而自动驾驶也许是一个没有起点的例子

first-visit：在一段episode中，一个状态只出现一次，或者只需计算第一次的价值

![](http://s8.sinaimg.cn/middle/002RSgYjzy7evWgksxF77&690)

every-visit：在一段episode中，一个状态可能会被访问多次，需要计算每一次的价值

![](http://s9.sinaimg.cn/middle/002RSgYjzy7evWgks3e18&690)

on-policy method：评估和优化的策略和模拟的策略是同一个

![](http://s4.sinaimg.cn/middle/002RSgYjzy7evWgkAp583&690)

off-policy method：评估和优化的策略和模拟的策略是不同的两个。有时候，模拟数据来源于其他处，比如已有的数据，或者人工模拟等。

#### Incremental Implementation

![](http://s8.sinaimg.cn/middle/002RSgYjzy7evWgkGZ957&690)

#### Off-policy Monte Carlo Control

![](http://s16.sinaimg.cn/middle/002RSgYjzy7evWgkGNV9f&690)

target policy：目标策略，off policy method中，需要优化的策略。

behavior policy：行为策略，off policy method中，模拟数据来源的策略。

根据上面的不同情境，在强化学习中，提供了不通的蒙特卡洛方法：

- 蒙特卡洛（exploring starts）方法
- on-policy first-visit蒙特卡洛方法
- off-policy every-visit蒙特卡洛方法

### 蒙特卡洛方法和动态规划的区别

1.动态规划是基于模型的，而蒙特卡洛方法是无模型的

	基于模型（model-base）还是无模型（model-free）的区别就是看（状态或者行动）价值（G,v(s),q(s,a)）是如何获得的
	
	如果是已知，根据已知的数据计算出来的，就是基于模型的
	如果是取样得到的，试验得到的就是无模型的

2.动态规划方法的计算是引导性的（bootstarpping），而蒙特卡洛方法的计算是取样性的（sampling）

	从计算状态价值V(s),q(s,a)的过程来看，动态规划是从初始状态开始，一次计算一步可能发生的所有状态价值，然后迭代计算下一步的所有状态价值。这就是引导性。

	蒙特卡洛方法是从初始状态开始，通过在实际环境中模拟，得到一段episode，比如结束是失败，这段episode上的状态节点本次价值都为0，如果成功了，本次价值都为1。

#### Off-policy Prediction via Importance Sampling

所有的学习控制方法都面临一个两难：需要从最优的behavior中学习action values，但是又要表现不那么最优来探索所有可能的行动。on-policy饿学习过程实际上是一个妥协：它不是为了最优policy来学习action values，而是学习一次次优的policy来保持一定的探索。一个直接的办法就是用两个policy，一个是保持探索性来生成behavior，称之为behavior policy，另外一个从这些behavior中学习target policy。这个过程就是off-policy learning。

on-policy方法通常都是更加简单和优先考虑使用的。off-policy方法涉及到不同的policy，通常都有更大的variance也更难以收敛。但是，off-policy更加强大和通用。
	
### 蒙特卡洛方法的优势

- 蒙特卡洛方法可以从交互中直接学习优化的策略，而不需要一个环境的动态模型。所谓环境的动态模型，就是表示环境的状态变化是可以完全推导的，表明要了解环境的所有知识。而蒙特卡洛可以计算v(s),q(s,a)不需要了解所有状态变化的可能性，只需要一些取样就可以。

- 蒙特卡洛方法可以用于模拟模型。

- 蒙特卡洛方法可以只考虑一个小的状态子集。

- 蒙特卡洛的每个状态价值计算是独立的，不会影响其他的状态价值。

### 蒙特卡洛方法的劣势

- 需要大量的探索（模拟）
- 基于概率的，不是确定性的
	
#### Summary
Monte Carlo methods代表着从所有情景中进行采样一些经验来学习value functions和最优policies。相对于DP methods给出至少三个优势。首先，能与环境直接交互来学习最优的behavior，而不需要环境的动态模型。其次，MC能够用模拟的或者采样模型的数据。令人惊喜的是，众多应用场景能够轻松模拟出采样场景，尽管这样的场景比较难以构建那种DP methods所需的转移概率的明确的模型。最后，MC能轻松有效的聚焦于一个小的状态子集。

其实，MC还有第四个优势，那就是更少的受到违反马尔科夫性质的伤害。This is because they do not update their value estimates on the basis of the value estimates of successor states,in other words,it is because do not bootstrap.

GPI involves interacting processes of policy evaluation and policy improvement. MC methods provide an alternative policy evaluation process.Rather than use a model to compute the value of each state,they simply average many returns that start in the state.Because a state's value is the expected return,this average can become a good approximation action-value functions,because these can be used to improve the policy without requiring a model of the environment's transition dynamics.MC methods intermix policy evaluation and policy improvement steps on an episode-by-episode basis,and can be incrementally implemented on an episode-by-episode basis.

保持有效的exploration是MC control methods的一个问题。MC control methods不能够刚好去选择那些一般估计到最好的actions，因为对于这些可选的actions，将不会有returns获取到，对于这些actions，即便其实际上表现更好，但也有可能从来不会学习到。一个解决办法就是直接忽略这个问题，通过假设这个场景从随机的state-action pairs开始，从其中选择覆盖所有可能性的actions。

## 时序差分学习（Temporal-Difference Learning）

时序差分学习结合了动态规划和蒙特卡洛方法

- 蒙特卡洛的方法是模拟（or经历）一段episode，在episode结束后，根据episode上各个状态的价值来估计状态价值。
- 时序差分的方法是模拟（or经历）一段episode，每行动一步（or几步），根据新状态的价值，然后估计执行前的状态价值。
- 简单认为就是蒙特卡洛是最大步数的时序差分学习。

如果可以计算出策略价值V(s),或者行动价值q(s,a),就可以优化策略π，在蒙特卡洛方法中，计算策略的价值，需要完成一个episode，通过情节的目标价值Gt来计算状态的价值

	V(St) <- V(St) + αδt
	δt = [Gt - V(St)]
	where
	δt  - Monte Carlo error
	α   - learning step size

时序差分的思想是通过下一个状态的价值计算状态的价值，形成一个迭代公式：

	V(St) <- V(St) + αδt
	δt = [Rt+1  + γV(St+1) - V(St)]
	where
	δt  - TD error
	α   - learning step size
	γ   - reward discount rate

- 单步时序差分学习方法TD(0)

![](http://s13.sinaimg.cn/middle/002RSgYjzy7eerawjWk2c&690)

- 多步时序差分学习方法

![](http://s16.sinaimg.cn/middle/002RSgYjzy7eeraF8p9af&690)


## 规划型方法和学习型方法（Planning and Learning with Tabular Methods）

observation的model，agent通过model来预测action的反应。

对于随机的observation，有两种不同的model：

- distribution model，分布式模型，返回行为的各种可能和其概率，不连续的action值。
- sample model，样本式模型，根据概率，返回行为的一种可能。
	
	其数学表达：(R,S') = model(S,A)

两种model的学习方法：

- planning methods 规划型方法，通过模型来获得价值信息（行动状态转换，奖赏等）。比如动态规划（dynamic programming）和启发式查询（heuristic search）。

- learning methods 学习型方法，通过体验（experience）来获得价值信息。比如蒙特卡洛方法（Mento Carlo method）和时序差分方法（temporal different method）。

规划型方法和学习型方法都是通过计算策略价值来优化策略。因此，可以融合到一起。

#### Dyna 结合模型学习和直接强化学习

- model learning 模型学习，通过体验来优化模型的过程
- directly reinforcement learning 直接强化学习，通过体验来优化策略的过程

### Tabular Dyna-Q

	Initialize Q(s,a) and Model(s,a) under conditons of s∈S and a∈A(s)
	Do forever(for each episode):
		(a) S <- current(nonterminal) state
		(b) A <-  ε-greedy(S,Q)
		(c) Execute action A;observe resultant reward,R,and state,S’
		(d) Q(S,A) <- Q(S,A) + α[R+γmaxQ(S’,a)-Q(S,A)]
		(e) Model(S,A) <- R,S’ (assuming deterministic environment)
		(f) Repeat n times:
			S <- random previously observed state
			A <- random action previously taken in S
			R,S’ <- Model(S,A)
			Q(S,A) <- Q(S,A) + α[R+γmaxQ(S’,a)-Q(S,A)]

上面的算法，如果n=0，就是Q-Learning。Dyna-Q的算法的优势在于性能上的提高。主要原因是通过建立model，减少了(c)的操作，模型学习到了Model(S,A) <- R,S'

### Prioritized Sweeping
提供了一种性能上的优化算法，只评估那些误差大于一定值θ的策略价值。


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


## on-policy 预测的近似方法

找到一个对策略的状态价值通用的的近似预测方法V^(s,θ)

如何判断这种近似方法好不好呢？

价值均方误差(Mean Squared Value Error)

![](http://s6.sinaimg.cn/middle/002RSgYjzy7eexcnSlL85&690)

如何求θ呢？一个常见的方法就是通过梯度递减的方法，迭代求解θ

- 随即梯度递减方法（Stochastic gradient descend method）

	这个方法可以在多次迭代后，让θ最优

- 蒙特卡洛

![](http://s7.sinaimg.cn/middle/002RSgYjzy7eexW5j9456&690)

- 半梯度递减方法（Semi-gradient method）
	
	TD(0)和n-steps TD计算价值的公式不是精确的，而蒙特卡洛方法是精确的。

![](http://s7.sinaimg.cn/middle/002RSgYjzy7eexW5rM2e6&690)


## on-policy控制的近似方法

近似控制方法（Control Methods）是求策略的行动状态价值q(s,a)的近似值q^(s,a,θ)

- 半梯度递减的控制Sarsa方法（Episode Semi-gradient Sarsa for Control）
- 多步半梯度递减的控制Sarsa方法(n-step Semi-gradient Sarsa for Control)
- 半梯度递减Sarsa的平均奖赏版(for continuing tasks)
- 多步半梯度递减的控制Sarsa方法 - 平均奖赏版(for continuing tasks)

## off-policy的近似方法

off-policy的近似方法的研究处于领域的前沿，主要有两个方向：

- 使用重要样本的方法，扭曲样本的分布成为目标策略的分布。这样就可以使用半梯度递减方法收敛。
- 开发一个真正的梯度递减方法，这个方法不依赖于任何分布。


## Model-Free Control（解决未知Environment下的Control问题）

	解决未知policy情况下未知Environment的MDP问题，也就是Model-Free Control问题，这是最常见的强化学习问题。
	
	-On-Policy Monte-Carlo
	
	动态规划解决planning问题（已知Environment）中提出policy iteration和value iteration，其中policy itertion和policy improvenment组成。
	
	未知Environment的policy evaluation是通过蒙特卡洛方法求解，结合起来得到一个解决Model-Free control方法，先通过贪婪算法来确定当前的policy，
	再通过蒙特卡洛policy evaluation来评估当前的policy好不好，再更新policy。

	在已知Environment情况下，policy improvement更新的解决办法是通过状态转移矩阵把所有可能转移到的状态得到的价值函数值都计算出来，选出最大的。
	但未知environment没有状态转移矩阵，因此只能通过最大化动作价值函数来更新policy，由于improvement的过程需要动作值函数，那么在policy evaluation
	的过程中针对给定的policy需要计算的回报价值函数V(s)也替换成动作值函数Q(s,a)。
		
![](http://s11.sinaimg.cn/middle/002RSgYjzy7dm7BYFtg4a&690)
	
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
			Q(S,A) = Q(S,A)+α(R+γQ(S',A')-Q(S,A))
		- ====更新状态S=S'
		- ==直到S到达终止状态

	-Off-Policy Learning

	Off-Policy Learning是在某个已知策略（behaviour policy）μ(a|s) 下来学习目标策略(target policy)π(a|s),这样就可以从人的控制或者其他表现的
	比较好的agent中来学习新的策略。

	已知策略分布P(X),目标策略分布Q(X),reward函数f(X),两种分布中reward期望为Exp[f(X)],从μ中来估计π获得的return，此方法称为Importance Sampling。

	同样的Off-Policy TD也是改变了更新值函数公式，改变的这一项相当于给TD target加权，这个权重代表了目标策略和已知策略匹配程度，代表了是否能够信任目标
	policy提出的这个action。

![](http://s7.sinaimg.cn/middle/002RSgYjzy7dm8KD1I256&690)
		
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


##  策略梯度方法（Policy Gradient Methods）

In  policy graident methods,the policy can be parameterized in any way ,as long as π(a|s,θ) is differentiable with respect to its parameters,that is ,as long as ▽π(a|s,θ) exists and is always finite.

Polocy-based methods also offer useful ways of dealing with continuous action spaces.


#### 策略梯度方法的新思路

![](http://s9.sinaimg.cn/middle/002RSgYjzy7efJOHtzie8&690)

π(a|s,θ)其输出的是在状态s下，选择动作a的概率

#### 策略梯度定理 （The policy gradient theorem）

情节性任务

如何计算策略的价值η

		η(θ) ≈ vπθ(s0)
		where 
		η the performance measure
		vπθ the true function for πθ，the policy determined by θ
		s0 some particular state
		
			- 策略梯度定理
			▽η(θ) = Σdπ(s)Σqπ(s,a)▽θπ(a|s,θ)
			where
			d(s)  on-policy distribution,the fraction of time spent in s under the target policy π
			Σd(s) = 1
			
			dπ(s) is defined to be the expected number of time steps t on which St=s in a randomly generated episode starting in s0 and following π and the dynamics of the MDP.


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

##### REINFORCE:Monte-Carlo Policy Gradient

The policy gradient theorem gives us an exact expression for this gradient,all we need is some way of sampling whose expectation equals of approximates this expression.

策略价值计算公式

![](http://s9.sinaimg.cn/middle/002RSgYjzy7enGgEgHef8&690)

Update Rule公式

![](http://s13.sinaimg.cn/middle/002RSgYjzy7enGgEhmAfc&690)

目标是优化reward，也就是优化值函数，只是这里θ不是值函数的参数，而是policy的参数，如果目标函数对参数求导，得到policy的gradient的形式为：

![](http://s4.sinaimg.cn/middle/002RSgYjzy7dpknZ1R1a3&690)	
	
第一项是衡量policy朝当前选择（某个状态+某个动作）偏移的程度，第二项衡量了当前选择的好坏。

从而推导出Monte-Carlo Policy Gradient的形式，首先更新参数的方法是随机梯度下降+policy gradient，gradient中的动作值函数取值用执行过程中的return来代替。

作为一个随机梯度方法，REINFORCE有比较好的理论上的收敛性质。the expected update over an episode is in the same direction as the performance graident.this assures an improvement in expected performance for sufficiently small α,and vonvergence to a local optimum under standard stochastic approximation conditions for decreasing α.However, as a Monte-Carlo method REINFORCE may be of high variance and thus slow to learn.

##### REINFORCE with baseline:Monte-Carlo Policy Gradient

The policy gradient theorem can be generalized to include a comparison of the action value to an arbitrary baselien b(s).The baseline can be any function ,even a random variable,as long as it does not vary with a;the equation remains ture,because the subtracted quanlity is zero.

after we convert the policy gradient theorem to an expectation and an update rule,using the same steps as in the previous section,then the baseline can have a significant effect on the variance of the update rule.

For MDPs the baseline should vary with state.In some states all actions have high values and we need a high baseline to differentiate the higher valued actions form the less highly valued ones;in other states all actions will have low values and a low baseline si appropriate.

策略价值计算公式

![](http://s15.sinaimg.cn/middle/002RSgYjzy7enHCYVRkae&690)

Update Rule公式

![](http://s5.sinaimg.cn/middle/002RSgYjzy7enHCYT88b4&690)

## Actor-Critic

Methods that learn approximations to both policy and value functions are often called actor-critic methods,where actor is a reference to the learned policy,and cirtic refers to the learned value function,usually a state-value function.

此算法实际上是：

- 带baseline的蒙特卡洛策略梯度强化算法的TD通用化
- 加上有效跟踪（eligibility traces）

	*蒙特卡洛方法要求必须完成当前的episode，这样才能计算正确的回报Gt*
	
	*TD避免了这个条件（从而提高了效率），可以通过临时差分计算一个近似的回报Gt(0)≈Gt,当然也没那么精确*
	
	*有效跟踪(eligibility traces)优化了(计算权重变量的)价值函数的微分值 et ≈ ▽v^(St,θt) + γλet-1*

Although the REINFORCE-with-baseline method learns both a policy and a state-value function,we do not consider it to be an actor-critic method because its state-value function is used only as a baseline,not as a critic.That is,it is not used for bootstrapping(updating a state from the estimated values onf subsequent states),but only as a baseline for the state being updated.

As we hvae seen,the bias introduced through bootstrapping and reliance on the state representation is often on balance beneficial because it reduces variance and accelerates learning.REINFORCE with-baseline is unbiased and will converge asymptotically to a local minimum,but like all Monte-Carlo methods it tends to be slow to learn(of high variance) and inconvenient to implement online or for continuing problems.

With temporal-difference methods we can eliminate these inconveniences and through multi-step mehtods we can flexibly choose the degree of bootstrapping.In order to gain these advantages in the case of policy gradient methods we use actor-critic methods with a true bootstrapping critic.


Update Rule公式

![](http://s6.sinaimg.cn/middle/002RSgYjzy7enS6acQJ05&690)


Upate Rule with eligibility traces公式

![](http://s9.sinaimg.cn/middle/002RSgYjzy7enS6aikU68&690)


##### Actor-Critic Policy Gradient

	Monte-Carlo Policy Gradient是用episode中反馈的return当作是动作值函数的采样，如果采用value fcuntion approximation的方法，即迭代更新policy又更新值函数。


##### Policy Gradient for Continuing Problems

For continuing problems without episode boundaries we need to define performance in terms of the average rate of reward per time step.

策略价值计算公式
对于连续性任务的策略价值是每个步骤的平均奖赏

![](http://s15.sinaimg.cn/middle/002RSgYjzy7enVqiS50be&690)

Update Rule公式

![](http://s16.sinaimg.cn/middle/002RSgYjzy7enVqiVeT2f&690)

Update Rule Actor-Critic with eligibility traces(contunuing)公式

![](http://s2.sinaimg.cn/middle/002RSgYjzy7enVqiXPr11&690)

###### Summary

Parameterized policy methods also have an important theoretical advantage over action-value methods in the form of the form of the policy gradient theorem,which gives an exact formula for how performance is affected by the policy parameter that does not involve derivatives of the state distribution.This theorem provides a theoretical foundation for all policy gradient methods.

The REINFORCE method follows directly from the policy gradient theorem.Adding a state-value function as a baseline reduces REINFORCE's variance without introducing bias.Using the state-value function for bootstrapping results introduces bias,but is often desirable for the same reason that bootstrapping TD methods are often superior to Monte Carlo methods(substantially reduced variance).The state-value function assigns credit to the policy's action selections,and accordingly the former is termed the critic and the latter the actor,and these overall methods are sometimes termed actor-critic methods.

Overall,policy-gradient methods provides a significantly different set of proclicities,strengths,and weaknesses than action-value methods.

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



## 有效跟踪（Eligibility Traces）

别人翻译成资格迹，反正都是不好理解的东西，在数学上，其就是一个向量，称为eligibility trace vector

强化学习就是找最优策略π

最优策略等价于最优行动π(s)

最优行动可以由最优状态价值v(s)或者最优行动价值q(s,a)决定

强化学习可以简单理解成求这个v(s)和q(s,a)

在近似方法中V(s)或者q(s,a)表示为近似预测函数v^(s,θ)或者近似控制函数q^(s,a,θ）

求近似预测函数V*(s,θ),就是求解权重向量θ

求权重向量θ是通过梯度下降的方法，比如：

	δt = Gt - V^(St,θt)
	θt+1 = θt + αδt*▽(v^(St,θt))

α，Gt，▽v^(St,θt)，每个都有自己的优化方法。

其中▽(v^(St,θt)可以通过有效跟踪来优化，有效跟踪就是优化后的函数微分，之所以要优化，原因在于TD算法中V^(St,θt)是不精确的，Gt也是不精确的。

δt是误差，权重向量θt+1就是在θt的基础上，加上αδt*有效跟踪。

#### et  
第t步的有效跟踪向量（eligibility trace rate）

有效跟踪向量是近似价值函数的优化微分值

其优化的技术称为(backward view)

#### On-line Forward View

on-line和off-line的一个区别是off-line的数据是完整的，比如拥有一个episode的所有Return (G)。