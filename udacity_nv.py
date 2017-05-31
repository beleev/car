import argparse
import h5py
import tensorflow as tf
import numpy as np
import random
import time
import sys
import cv2
import csv

from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

FLAGS = None
path = './simdata/'
labels_file = './simdata/driving_log.csv'

row = 66
col = 200

class DataSet(object):

    def __init__(self):

        self.label = []
        with open(labels_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                self.label.append(line)
            self.label.pop(0)
        self.num = len(self.label)
        self.leftpool = range(self.num)

    def get_image(self, img_path):
        img = plt.imread(img_path)
        return cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(col,row))

    def readrandom(self, size):
        if len(self.leftpool) < size:
            return self.readrandom(len(self.leftpool))
        else:
            image_list = []
            label_list = []
            for i in range(size):
                index = random.choice(range(len(self.leftpool)))
                sel = self.leftpool.pop(index)
                for j in range(3):
                    img_path = self.label[sel][j]
                    img_path = path + img_path.strip()

                    image_list.append(self.get_image(img_path))
                    if j == 1:
                        nlabel = float(self.label[sel][3]) + 0.2
                    elif j == 2:
                        nlabel = float(self.label[sel][3]) - 0.2
                    else:
                        nlabel = float(self.label[sel][3])
                    label_list.append(nlabel)
            images = np.array(image_list).astype(np.float32)
            labels = np.array(label_list).astype(np.float32)
            images = np.append(images,images[:,:,::-1],axis=0)
            labels = np.append(labels,-labels,axis=0)

            images = images.reshape(-1, row, col, 1)
            labels = labels.reshape(-1, 1)
            if len(self.leftpool) == 0:
                self.leftpool = range(len(self.label))
            return images, labels


class Trainer(object):

    def __init__(self):
        self.learning_rate = 0.001
        self.dropout = 0.5

        self.batch_size = 128
        self.epoch = 10
        self.display_step = 1

        self.data = DataSet()
        self.training_iters = self.data.num * self.epoch

    def conv2d(self, x, W, b, strides=1, mode='SAME'):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=mode)
        return tf.nn.bias_add(x, b)

    def pool2d(self, x, k=2, mode='VALID'):
        return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=mode)

    def conv_net(self, x, weights, biases, dropout):
        nor = tf.add(tf.multiply(x, 1/127.5), -1)
        conv1 = self.conv2d(nor, weights['wc1'], biases['bc1'])
        conv1 = self.pool2d(conv1, k=2, mode='SAME')
        conv1 = tf.nn.elu(conv1)

        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.pool2d(conv2, k=2, mode='SAME')
        conv2 = tf.nn.elu(conv2)

        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'], mode='VALID')
        conv3 = self.pool2d(conv3, k=2, mode='VALID')
        conv3 = tf.nn.elu(conv3)

        conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'], mode='VALID')
        conv4 = tf.nn.elu(conv4)

        conv5 = self.conv2d(conv4, weights['wc5'], biases['bc5'], mode='VALID')
        conv5 = tf.nn.elu(conv5)

        fc1 = tf.nn.dropout(conv5, dropout)
        fc1 = tf.reshape(fc1, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.elu(fc1)

        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.elu(fc2)
        fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
        fc3 = tf.nn.elu(fc3)

        out = tf.nn.dropout(fc3, dropout)
        out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
        return out

    def build_net(self):
        self.x = tf.placeholder(tf.float32, [None, row, col, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        self.weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 24], stddev=0.1)),
            'wc2': tf.Variable(tf.random_normal([5, 5, 24, 36], stddev=0.1)),
            'wc3': tf.Variable(tf.random_normal([5, 5, 36, 48], stddev=0.1)),
            'wc4': tf.Variable(tf.random_normal([3, 3, 48, 64], stddev=0.1)),
            'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.1)),
            # fully connected, w*h*64 inputs, 1 outputs
            'wd1': tf.Variable(tf.random_normal([2*19*64, 100], stddev=0.1)),
            'wd2': tf.Variable(tf.random_normal([100, 50], stddev=0.1)),
            'wd3': tf.Variable(tf.random_normal([50, 10], stddev=0.1)),
            # out
            'out': tf.Variable(tf.random_normal([10, 1], stddev=0.1))
        }

        self.biases = {
            # conv
            'bc1': tf.Variable(tf.random_normal([24])),
            'bc2': tf.Variable(tf.random_normal([36])),
            'bc3': tf.Variable(tf.random_normal([48])),
            'bc4': tf.Variable(tf.random_normal([64])),
            'bc5': tf.Variable(tf.random_normal([64])),
            # full conn
            'bd1': tf.Variable(tf.random_normal([100])),
            'bd2': tf.Variable(tf.random_normal([50])),
            'bd3': tf.Variable(tf.random_normal([10])),
            # out
            'out': tf.Variable(tf.random_normal([1]))
        }

        self.pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.square(self.pred - self.y))

    def run(self, model_path=None, save_path=None):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            if model_path:
                saver.restore(sess, model_path)
                print("Model restored from file: %s" % model_path)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            step = 1
            # Keep training until reach max iterations
            while step * self.batch_size <= self.training_iters:
                batch_x, batch_y = self.data.readrandom(self.batch_size)
                sess.run(optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})
                if step % self.display_step == 0:
                    loss = sess.run(self.cost, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '    ' +\
                          "Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + "{:.16f}".format(loss))
                    #pred = sess.run(self.pred, feed_dict={self.x: batch_x, self.keep_prob: 1.})
                    #print pred.reshape(1,-1)
                step += 1
            if save_path:
                save_model = saver.save(sess, save_path)
                print("Model saved in file: %s" % save_model)

    def predict(self, model_path, image_path):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_path)
            print("Model restored from file: %s" % model_path)
            img = plt.imread(image_path)
            img = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(col,row))
            img = np.multiply(img.astype(np.float32).reshape(-1, row, col, 1), 2.0 / 255.0) - 1
            pred = sess.run(self.pred, feed_dict={self.x: img, self.keep_prob:1.})
            print pred.reshape(1)[0]


def main(unused_argv):
    model = Trainer()
    model.build_net()
    if FLAGS.predict:
        model.predict(model_path=FLAGS.model_path, image_path=FLAGS.predict)
        print("Prediction Done!")
    else:
        model.run(model_path=FLAGS.model_path, save_path=FLAGS.save_path)
        print("Training Done!")


if __name__ == '__main__':
    #import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Directory to load model'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Directory to save model'
    )
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help='image to predict'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

