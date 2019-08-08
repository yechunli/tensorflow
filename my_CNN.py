import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

class neural_network():
    def __init__(self, Num_input, Num_FNN_layers, Num_features, Num_hidden, feature_size, Num_output, batch_size, image_width, image_height, Num_CNN_layers, max_pool_size):
        self.Num_input = Num_input
        self.Num_hidden = Num_hidden
        self.Num_output = Num_output
        self.Num_FNN_layers = Num_FNN_layers
        self.Num_features = Num_features
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.Num_CNN_layers = Num_CNN_layers
        self.max_pool_size = max_pool_size
        if Num_CNN_layers != 0:
            self.data = tf.placeholder(tf.float32, shape=[self.batch_size, 28, 28, 1])
            self.label = tf.placeholder(tf.int32, shape=[self.batch_size])
        else:
            self.data = tf.placeholder(tf.float32, shape=[self.batch_size, 784])
            self.label = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = 0.01

        self.build_model()

    def CNN_init(self):
        conv_weight = []
        conv_bias = []
        for i in range(self.Num_CNN_layers):
            conv_weight.append(tf.Variable(tf.truncated_normal([self.feature_size[i],self.feature_size[i],self.Num_features[i],self.Num_features[i+1]],stddev=0.1)))
            conv_bias.append(tf.Variable(tf.zeros([self.Num_features[i+1]])))
        return conv_weight, conv_bias

    def FNN_init(self):
        full_weight = []
        full_bias = []
        if self.Num_CNN_layers != 0:
            input_size = (self.image_height / (2**(self.Num_CNN_layers))) ** 2
            input_size *= self.Num_features[-1]
        else:
            input_size = self.image_height * self.image_width
        if self.Num_FNN_layers == 0:
            output_weight = tf.Variable(tf.truncated_normal([input_size, self.Num_output],stddev=0.1))
            output_bias = tf.Variable(tf.zeros([self.Num_output]))
            #output_bias = tf.Variable(tf.truncated_normal([self.Num_output], stddev=0.1, dtype=tf.float32))
            full_weight.append(output_weight)
            full_bias.append(output_bias)
        else:
            input_weight = tf.Variable(tf.truncated_normal([tf.to_int32(input_size), self.Num_hidden[0]],stddev=0.1))
            input_bias = tf.Variable(tf.zeros([self.Num_hidden[0]]))
            full_weight.append(input_weight)
            full_bias.append(input_bias)
            for i in range(self.Num_FNN_layers-1):
                full_weight_temp = tf.Variable(tf.truncated_normal([self.Num_hidden[i+1], self.Num_hidden[i+1]],stddev=0.1))
                full_bias_temp = tf.Variable(tf.zeros([self.Num_hidden[i+1]]))
                full_weight.append(full_weight_temp)
                full_bias.append(full_bias_temp)
            output_weight = tf.Variable(tf.truncated_normal([self.Num_hidden[self.Num_FNN_layers-1],self.Num_output],stddev=0.1))
            output_bias = tf.Variable(tf.zeros([self.Num_output]))
            full_weight.append(output_weight)
            full_bias.append(output_bias)
        return full_weight, full_bias

    def dropout(self, x):
        return tf.nn.dropout(x, self.keep_prob)



    def CNN_layer(self, input_data):
        conv_weight, conv_bias = self.CNN_init()
        conv_result = input_data
        for i in range(self.Num_CNN_layers):
            conv = tf.nn.conv2d(conv_result, conv_weight[i],strides=[1,1,1,1],padding='SAME')
            conv_add = tf.nn.bias_add(conv, conv_bias[i])
            conv_relu = tf.nn.relu(conv_add)
            conv_result = tf.nn.max_pool(conv_relu,ksize=[1,self.max_pool_size[i],self.max_pool_size[i],1],strides=[1,self.max_pool_size[i],self.max_pool_size[i],1], padding='SAME')
            #dropout
            #batch nomalization
        output_dim = tf.shape(conv_result)
        output = tf.reshape(conv_result,[output_dim[0], output_dim[1]*output_dim[2]*output_dim[3]])
        return output

    def FNN_layer(self, input_data):
        full_weight, full_bias = self.FNN_init()
        full_result = input_data
        for i in range(len(full_weight)):
            full = tf.matmul(full_result, full_weight[i])
            full_add = tf.add(full, full_bias[i])
            if i == len(full_weight)-1:
                full_result = tf.identity(full_add)
            else:
                full_result = tf.nn.relu(full_add)
        return self.dropout(full_result)

    def loss_function(self, input_data, label):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=input_data))
        optimizer_op = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer_op.minimize(self.loss)
        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.loss)

    def get_accuracy(self, input, label):
        result = tf.argmax(input, axis=1)
        convert32 = tf.to_int32(result)
        temp = tf.equal(convert32, label)
        true_sum = tf.where(temp)
        count = tf.shape(true_sum)[0]
        #count = tf.reduce_sum(tf.equal(result, label))
        self.acc = 100. * tf.to_float(count)/self.batch_size
        with tf.name_scope('accuracy'):
            tf.summary.scalar('acc', self.acc)

    def build_model(self):

        if self.Num_CNN_layers != 0:
            CNN_output = self.CNN_layer(self.data)
            FNN_output = self.FNN_layer(CNN_output)
        else:
            FNN_output = self.FNN_layer(self.data)

        self.loss_function(FNN_output, self.label)
        self.get_accuracy(FNN_output, self.label)

data_dir = 'F:\python_project\mnist'
mnist = read_data_sets(data_dir)

batch_size = 100
generations = 500

Num_CNN_layers = 2
feature_size = [4, 4]
Num_features = [1, 25, 50]
max_pool_size = [2,2]

Num_FNN_layers = 1
Num_hidden = [100]
Num_output = 10


if Num_CNN_layers != 0:
    train_data = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
    print(train_data)
    train_label = mnist.train.labels
else:
    train_data, train_label = mnist.train.images, mnist.train.labels

if Num_CNN_layers != 0:
    test_data = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
else:
    test_data = mnist.test.images
test_label = mnist.test.labels

keep_pro = 0.9

def feed_dict(data ,label, keep_pro=1):
    index = np.random.choice(len(data), batch_size)
    original_data = data[index]
    input_data = np.expand_dims(original_data, 3)
    input_label = label[index]
    return {Network.data:input_data, Network.label:input_label, Network.keep_prob:keep_pro}


Network = neural_network(0,Num_FNN_layers, Num_features, Num_hidden, feature_size, Num_output, batch_size, 28,28, Num_CNN_layers, max_pool_size)
init = tf.initialize_all_variables()
merge = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('F:\python_project\log\my_CNN', sess.graph)
    for i in range(generations):
        summary, _ = sess.run([merge, Network.train], feed_dict=feed_dict(train_data, train_label, keep_pro))
        if (i+1) % 10 == 0:
            train_loss, train_acc = sess.run([Network.loss, Network.acc], feed_dict=feed_dict(train_data, train_label, keep_pro))
            print('%d :train_loss:%.2f, train_acc:%.2f' %(i/10, train_loss, train_acc))
            test_loss, test_acc = sess.run([Network.loss, Network.acc], feed_dict=feed_dict(test_data, test_label))
            print('%d :test_loss:%.2f, test_acc:%.2f' %(i/10, test_loss, test_acc))
            writer.add_summary(summary, int(i/10))
    writer.close()



