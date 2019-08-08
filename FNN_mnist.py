import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

max_step = 1000
learning_rate = 0.001
keep_out = 0.9

data_dir = 'F:\python_project\mnist'
log_dir = 'F:\python_project\log'


mnist = input_data.read_data_sets(data_dir, one_hot=True)


sess = tf.InteractiveSession()

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='label')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

def weight_init(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)

def bias_init(shape):
    init = tf.constant(0.1, shape = shape)
    return tf.Variable(init)

def variable_summary(var):
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input, Num_input, Num_hidden, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope(layer_name+'_weight'):
            weight = weight_init([Num_input, Num_hidden])
            variable_summary(weight)
        with tf.name_scope(layer_name+'_bias'):
            bias = bias_init([Num_hidden])
            variable_summary(bias)
        with tf.name_scope(layer_name+'_xW_plus_b'):
            pre_act = tf.matmul(input, weight) + bias
            tf.summary.histogram(layer_name+'_pre_act', pre_act)
        acted = act(pre_act)
        tf.summary.histogram(layer_name+'_acted', acted)
        return acted

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+'\\train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'\\test')

tf.global_variables_initializer().run()

def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = keep_out
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

for i in range(max_step):
    if i%10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s:%s' %(i, acc))
    else:
        if i%100 == 99:
            run_option = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            run_metada = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                  options=run_option, run_metadata=run_metada)
            train_writer.add_run_metadata(run_metada, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metdata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()