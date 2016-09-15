import tensorflow as tf
import numpy as np
import time

t = time.time()
# create data
x_data = np.random.rand(10000).astype(np.float)
y_data = x_data * 897.1 + 10000.23

#  create tensfowflow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # size is one element, range from -1 to 1
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # construct an optimizer
train = optimizer.minimize(loss)


init = tf.initialize_all_variables()
# structured finished

# define a session
#sess = tf.Session() #session is a class so capitalize S
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

sess.run(init) #always remember to run the sesion

for step in range(100000):
    sess.run(train)
        #     if step % 200 == 0:
        #         print(step, sess.run(Weights), sess.run(biases)) #session.run goes to execute command or access variables
        #         print sess.run(loss)

print time.time()-t
