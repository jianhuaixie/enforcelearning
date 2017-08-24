import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data+biases  # 预测的y
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 学习效率0.5
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()  # 初始化神经网络的结构图
### create tensorflow structure end ###
sess = tf.Session()
sess.run(init)  # 神经网络图激活

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))

