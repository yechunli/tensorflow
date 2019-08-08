import tensorflow as tf
import numpy as np

data_dir = 'F:\python_project\\tensorflow_neural\data.csv'
queue = tf.train.string_input_producer([data_dir])
reader = tf.TextLineReader()
key, value = reader.read(queue)
record_defaults = [[1], [1], [1]]
col1, col2, col3 = tf.decode_csv(value, record_defaults)
tmp_train = tf.stack([col1, col2])
tmp_label = [col3]

class FNN():
    def __init__(self, hidden_value, batch_size, learning_rate=0.01, keep_rate=1):
        self.train = tf.placeholder(tf.float32, [None,2])
        self.label = tf.placeholder(tf.float32, [None,1])
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_rate = keep_rate
        self.input_num = 2
        self.output_num = 1
        self.hidden_value = hidden_value
        self.hidden_layer_num = len(hidden_value)

        self.build_model()

    def variable_init(self, input_num, output_num):
        shape = [input_num, output_num]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        bias = tf.Variable(tf.zeros(shape=[output_num]))
        return weight, bias

    def drop_out(self, data):
        output = tf.nn.dropout(data, self.keep_rate)
        return output

    def layer_init(self):
        weight = []
        bias = []
        weight_layer1, bias_layer1 = self.variable_init(self.input_num, self.hidden_value[0])
        weight.append(weight_layer1)
        bias.append(bias_layer1)
        for i in range(self.hidden_layer_num-1):
            weight_hidden, bias_hidden = self.variable_init(self.hidden_value[i], self.hidden_value[i+1])
            weight.append(weight_hidden)
            bias.append(bias_hidden)
        weight_output, bias_output = self.variable_init(self.hidden_value[-1], self.output_num)
        weight.append(weight_output)
        bias.append(bias_output)
        return weight, bias

    def cal_loss(self, output, label):
        #loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output, name='loss')
        loss = tf.reduce_mean((output-label)**2)
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', loss)
        return loss

    def build_model(self):
        weight, bias = self.layer_init()
        train = tf.random_shuffle(self.train, seed=1)
        label = tf.random_shuffle(self.label, seed=1)
        #start_num = tf.random_uniform([],0, tf.shape(self.train)[0]-self.batch_size, dtype=tf.int32)
        start_num = 0
        label_batch = label[start_num: start_num+self.batch_size]
        next_layer_input = train[start_num: start_num+self.batch_size]
        for i in range(len(weight)):
            mul_tmp = tf.matmul(next_layer_input, weight[i])
            add_tmp = mul_tmp + bias[i]
            if i < len(weight) - 1:
                act_tmp = tf.nn.relu(add_tmp)
                drop_tmp = self.drop_out(act_tmp)
                next_layer_input = drop_tmp
            else:
                output = tf.nn.tanh(add_tmp)
                #output = add_tmp

        self.loss = self.cal_loss(output, label_batch)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
test = np.reshape([1, 1], [1,2])
test_label = [0]

def dict(fnn, data, label):
        return {fnn.train: data, fnn.label: label}

logdir = 'F:\python_project\log\\test'

coord = tf.train.Coordinator()
hidden_value = [8, 8]
batch_size = 10
fnn = FNN(hidden_value, batch_size)
generations = 1000
merge = tf.summary.merge_all()
train = []
label = []
init_variable = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_variable)
    writer = tf.summary.FileWriter(logdir, graph=sess.graph)
    thread = tf.train.start_queue_runners(sess, coord)
    while not coord.should_stop():
        for i in range(50):
            train_tmp, label_tmp = sess.run([tmp_train, tmp_label])
            train.append(train_tmp)
            label.append(label_tmp)
        coord.request_stop()
    coord.request_stop()
    coord.join(thread)
    for i in range(generations):
        _, loss, summary = sess.run([fnn.train_step, fnn.loss, merge], feed_dict=dict(fnn, train, label))
        # sess.run([fnn.train_step, fnn.loss], feed_dict=dict(fnn))
        if i % 10 == 0:
            # loss, summary = sess.run([fnn.loss, merge], feed_dict=dict(fnn, test, test_label))
            print(loss, i)
            writer.add_summary(summary, int(i / 10))
    writer.close()
