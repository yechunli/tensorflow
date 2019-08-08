import tensorflow as tf
import numpy as np

def shufflelists(lists):
    ri = np.random.permutation(len(lists[1]))
    out = []
    for l in lists:
        out.append(l[ri])
    return out

def hinder(hind_input, hind_num):
    row = hind_input.shape[1]
    W = tf.Variable(tf.truncated_normal([int(row), hind_num], stddev = 0.1), name = 'W')
    b = tf.Variable(tf.constant(0.1, shape=[hind_num]), name='b')
    pre_act = tf.matmul(hind_input, W) + b
    out = tf.nn.relu(pre_act, name='out')
    return out

D_input = 2
D_label = 1
D_hinder = 2
lr = 1e-4

x = tf.placeholder(tf.float32, [None, D_input], name='x')
t = tf.placeholder(tf.float32, [None, D_label], name='t')

# W_h1 = tf.Variable(tf.truncated_normal([D_input, D_hinder], stddev=0.1), name='W_h')
# b_h1 = tf.Variable(tf.constant(0.1, shape=[D_hinder]), name='b_h')
# pre_act_h1 = tf.matmul(x, W_h1) + b_h1
# act_h1 = tf.nn.relu(pre_act_h1, name='act_h')
h1 = hinder(x, 2)
h2 = hinder(h1, 8)
h3 = hinder(h2, 8)
h4 = hinder(h3, 8)
h5 = hinder(h4, 8)
result = hinder(h1, 1)


# W_o = tf.Variable(tf.truncated_normal([D_hinder, D_label], stddev=0.1), name='W_o')
# b_o = tf.Variable(tf.constant(0.1, shape=[D_label]), name='b_o')
# pre_act_o = tf.matmul(act_h1, W_o) + b_o
# y = tf.nn.relu(pre_act_o, name='act_y')

#loss = tf.reduce_mean((self.output-self.labels)**2)
loss = tf.reduce_mean((t-result)**2)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('int16')
Y = np.array(Y).astype('int16')

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

T = 10000

for i in range(T):
  sess.run(train_step,feed_dict={x:X,t:Y})

# batch_index = 0
# batch_size = 2
# for i in range(T):
#     X, Y = shufflelists([X, Y])
#     while batch_index< X.shape[0]:
#         sess.run(train_step, feed_dict={x:X[batch_index:(batch_index+batch_size)], t:Y[batch_index:(batch_size+batch_index)]})
#         batch_index += batch_size
print(sess.run(result, feed_dict={x:X}))

#Test = [[5, 5], [5, 100], [100, 200], [200, 200]]
#print(sess.run(result, feed_dict={x:Test}))
sess.close()