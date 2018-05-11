import tensorflow as tf
import os
import numpy as np
import shutil
import ex_utils
import warnings
import time
warnings.filterwarnings("ignore")

######################################################
MODEL_NAME = 'lotto10'
SAVE_DIR = "./output/{}_output/".format(MODEL_NAME)
BOARD_PATH = './board/{}_board'.format(MODEL_NAME)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

INPUT_DIM = 7
OUTPUT_DIM = 7
SEQ_LEN = 5
NLAYERS = 3
NUM_RNN_UNITS = 200
NUM_FC_UNITS = 100

TOTAL_EPOCH = 100001
BATCH_SIZE = 100
INIT_LEARNING_RATE = 0.1

trainX, trainY, testX, testY = ex_utils.create_lotto_data(SEQ_LEN, 999)

######################################################
#lotto_nets.py
def lstm_cell(rnn_hidden_units):
    cell = tf.contrib.rnn.LSTMCell(num_units=rnn_hidden_units, initializer = tf.glorot_uniform_initializer(seed=0), state_is_tuple=True)
    return cell

def linear_layer(x, input_dim, output_dim, stddev, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(seed=0, stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))
        h = tf.add(tf.matmul(x, W), b, name = 'h')
    return h

def tanh_layer(x, input_dim, output_dim, stddev, name):
    with tf.variable_scope(name):
        W =tf.get_variable('W', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(seed=0, stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))
        h = tf.nn.tanh(tf.add(tf.matmul(x, W), b), name = 'h')
    return h

def dropout_tanh_layer(x, input_dim, output_dim, stddev, keep_prob, name):
    with tf.variable_scope(name):
        W =tf.get_variable('W', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(seed=0, stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))
        h = tf.nn.tanh(tf.add(tf.matmul(x, W), b), name = 'h')
        h = tf.nn.dropout(h, keep_prob = keep_prob)
    return h

######################################################
#lotto_model.py
class Model:
    def __init__(self, sess):
        self.sess = sess
        self.num_rnn_units = NUM_RNN_UNITS
        self.num_fc_units = NUM_FC_UNITS
        self.model_name = MODEL_NAME
        self.input_dim = INPUT_DIM
        self.output_dim = OUTPUT_DIM
        self.nlayers = NLAYERS
        self.seq_len = SEQ_LEN
        self.board_path = BOARD_PATH
        self._build_model()

    def _build_model(self):
        tf.set_random_seed(0)
        with tf.variable_scope("Inputs") as scope:
            self.X = tf.placeholder(tf.float32, [None, self.seq_len, self.input_dim], name='X')
            self.Y = tf.placeholder(tf.float32, [None, self.output_dim], name='Y')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("LSTMLayer"):
                stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell(self.num_rnn_units) for _ in range(self.nlayers)])
                outputs, _states = tf.nn.dynamic_rnn(stacked_lstm, self.X, dtype=tf.float32)

            h1 = dropout_tanh_layer(outputs[:, -1], self.num_rnn_units, self.num_fc_units , stddev=0.01, keep_prob = self.keep_prob, name="FCLayer1")
            h2 = linear_layer(h1, self.num_fc_units, self.output_dim, stddev=0.01, name="FCLayer2")
            self.prediction = tf.identity(h2, name = 'prediction')

            with tf.variable_scope("Optimization"):
                self.cost = tf.reduce_mean(tf.square(self.Y-h2), name='cost')
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

            with tf.variable_scope("Scalars"):
                self.avg_loss = tf.placeholder(tf.float32)
                self.loss_scalar = tf.summary.scalar('loss', self.avg_loss)
                self.merged = tf.summary.merge_all()

            if os.path.exists(self.board_path):
                shutil.rmtree(self.board_path)

            self.writer = tf.summary.FileWriter(self.board_path)
            self.writer.add_graph(self.sess.graph)

    def predict(self, x_test):
        if len(x_test == 1):
            x_test = np.expand_dims(x_test, axis=0)
        p = self.sess.run(self.prediction, feed_dict={self.X: x_test, self.keep_prob : 1.0})
        p = (p*45).astype(int)
        return p

    def train(self, x_train, y_train, u):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.learning_rate: u, self.keep_prob : 0.7})

    def summary_log(self, avg_loss, epoch):
        s = self.sess.run(self.merged, feed_dict={self.avg_loss: avg_loss})
        self.writer.add_summary(s, global_step=epoch)

    def save(self, dirname):
        saver = tf.train.Saver()
        saver.save(self.sess, dirname)

######################################################
#main.py
with tf.Session() as sess:
    m = Model(sess)
    total_step = int(len(trainX)/BATCH_SIZE)
    print("The number of total steps : ", total_step)

    u = INIT_LEARNING_RATE
    for epoch in range(TOTAL_EPOCH):
        loss_per_epoch = 0
        epoch_start_time = time.perf_counter()

        np.random.seed(epoch)
        mask = np.random.permutation(len(trainX))
        trainX = trainX[mask]
        trainY = trainY[mask]
        for step in range(total_step):
            s = step * BATCH_SIZE
            t = (step + 1) * BATCH_SIZE
            minibatchX = trainX[s:t]
            minibatchY = trainY[s:t]
            c, _ = m.train(minibatchX, minibatchY, u)
            loss_per_epoch += c

        loss_per_epoch = loss_per_epoch/total_step
        m.summary_log(loss_per_epoch, epoch)
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        u = u*0.95
        if epoch%100 == 0:
            print("Epoch : [{:5d}/{:5d}] Loss : {:.6f} EpochDuration : {:.6f}(s)".format(epoch, TOTAL_EPOCH,loss_per_epoch,epoch_duration))
            print("=== 예측값 ===")
            print(m.predict(testX[0]))
            print("=== 실제값 ===")
            print((testY[0]*45).astype(int))

        m.save(SAVE_DIR + MODEL_NAME + "_{}/Model_LOTTO".format(epoch))