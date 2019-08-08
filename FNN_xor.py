import tensorflow as tf
import numpy as np

class FNN():
    def __init__(self, Num_input, Num_hidden, Num_output, Num_label, Num_hidden_layer, learning_rate=1e-3, l2_lambda=1e-2, dropout_rate=0.9):
        self.Num_input = Num_input
        self.Num_hidden = Num_hidden
        self.Num_output = Num_output
        self.Num_label = Num_label
        self.Num_hidden_layer = Num_hidden_layer
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        with tf.name_scope('Input'):
            self.input = tf.placeholder(tf.float32, [None, self.Num_input], name='input')
        with tf.name_scope('Label'):
            self.label = tf.placeholder(tf.float32, [None, self.Num_label], name='label')

        self.build()

    def weight_init(self, shape):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init)

    def bias_init(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def dropout(self, var, dropout_rate):
        return tf.nn.dropout(var, dropout_rate)

    def scalar_summary(self, name, var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name, mean)

    def layer(self, layer_name, Num_hidden, var, in_dim, act_function=tf.nn.relu):
        weight_shape = [in_dim, Num_hidden]
        bias_shape = [Num_hidden]
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name+'_weight'):
                weight = self.weight_init(weight_shape)
                self.W.append(weight)
                self.scalar_summary(layer_name+'\weight', weight)
            with tf.name_scope(layer_name+'_bias'):
                bias = self.bias_init(bias_shape)
                self.scalar_summary(layer_name+'\\bias', bias)
            with tf.name_scope(layer_name+'_XW_plus_b'):
                before_act = tf.matmul(var, weight) + bias
                tf.summary.histogram(layer_name+'\\before_act', before_act)
            with tf.name_scope(layer_name+'_activation'):
                activation = act_function(before_act)
                tf.summary.histogram(layer_name+'\\activation', activation)
            with tf.name_scope(layer_name+'_l2_loss'):
                self.l2_loss = tf.nn.l2_loss(weight)
        return self.l2_loss, activation

    def build(self):
        self.W = []
        self.loss = []
        self.loss_sum = tf.constant(0.0)

        l2_loss, temp = self.layer('input_layer', self.Num_hidden, self.input, self.Num_input)
        self.output1 = temp
        temp = self.dropout(temp, self.dropout_rate)
        self.loss.append(l2_loss)
        self.dropout_temp = temp
        # l2_loss, temp = self.layer('hidden_layer', self.Num_hidden, temp, self.Num_hidden)
        # temp = self.dropout(temp, self.dropout_rate)
        # self.loss.append(l2_loss)
        l2_loss, temp = self.layer('output_layer', self.Num_output, temp, self.Num_hidden, tf.identity)
        self.output = self.dropout(temp, self.dropout_rate)
        self.loss.append(l2_loss)

        with tf.name_scope('cross'):
            cross = tf.reduce_mean((self.output - self.label) ** 2)
            self.scalar_summary('cross', cross)

        with tf.name_scope('l2_loss'):
            for l2 in self.loss:
                self.loss_sum += l2
            cross_l2 = self.loss_sum * self.l2_lambda
            self.scalar_summary('l2_loss', cross_l2)

        with tf.name_scope('total_loss'):
            self.total_loss = cross + cross_l2
            self.scalar_summary('total_loss', self.total_loss)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

input = [[0, 0], [0, 1], [1, 0], [1, 1]]
label = [0, 1, 1, 0]
X = np.array(input).reshape([4,2])
Y = np.array(label).reshape([4,1])
neural_network = FNN(2, 10, 1, 1, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    merge = tf.summary.merge_all()

    writer = tf.summary.FileWriter('F:\python_project'+'\log\\train', graph=sess.graph)

    for i in range(10000):
        # W0 = sess.run(neural_network.W[0])
        # W1 = sess.run(neural_network.W[1])
        summary, output, _ = sess.run([merge, neural_network.output, neural_network.train_step], feed_dict={neural_network.input:X,
                                                                neural_network.label:Y})
        if i%100 == 0:
            #print('before\n', xx)
            #print('after\n', yy)
            # # 显示
            # print('W_0:\n%s' % sess.run(neural_network.W[0]))
            # print('W_1:\n%s' % sess.run(neural_network.W[1]))
            #print(output)
            writer.add_summary(summary, int(i/100))
    writer.close()
