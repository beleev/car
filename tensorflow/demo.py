import h5py
import tensorflow as tf
import numpy as np


path = '/root/car/data/'

class DataSet(object):
    
    def __init__(self, id_num):
        attr_path = path + str(id_num) + '/attr.h5'
        image_path = path + str(id_num) + '/image.h5'
        attr = h5py.File(attr_path, 'r')
        self.time = attr['attrs'][:,0]
        self.label = attr['attrs'][:,3]
        self.image = h5py.File(image_path, 'r')
        self.num = len(self.label)
        self.index = 0

    def read(self, size):
        if size + self.index > self.num:
            return read(self, self.num - self.index)
        else:
            begin = self.index
            end = self.index + size
            image_list = []
            for t in self.time[begin:end]:
                image_list.append(self.image["{:.3f}".format(t)])
            images = np.array(image_list)
            images = np.multiply(images.astype(np.float32), 1.0 / 255.0)
            images = images - 0.5
            labels = self.label[begin:end].reshape(-1, 1)
            if end == self.num:
                self.index = 0
            else:
                self.index = end 
        return images, labels
            
        
class Trainer(object):

    def __init__(self, id_num):
        self.learning_rate = 0.001
        self.dropout = 0.75

        self.training_iters = 3000
        self.batch_size = 3
        self.display_step = 1

        self.data = DataSet(id_num)
        
    def conv2d(self, x, W, b, strides=2):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return x
    
    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def conv_net(self, x, weights, biases, dropout):
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = self.maxpool2d(conv3, k=2)

        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc2 = tf.nn.dropout(fc1, dropout)
    
        fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
        fc3 = tf.nn.dropout(fc2, dropout)

        fc3 = tf.add(tf.matmul(fc3, weights['wd3']), biases['bd3'])
        fc3 = tf.nn.dropout(fc3, dropout)
        out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
        return out

    def build_net(self):
        self.x = tf.placeholder(tf.float32, [None, 320, 320, 3])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        weights = {
            # 5x5 conv, 3 input, 24 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
            'wc2': tf.Variable(tf.random_normal([3, 3, 24, 48])),
            'wc3': tf.Variable(tf.random_normal([3, 3, 48, 64])),
            # fully connected, w*h*64 inputs, 1 outputs
            'wd1': tf.Variable(tf.random_normal([1600, 1024])),
            'wd2': tf.Variable(tf.random_normal([1024, 128])),
            'wd3': tf.Variable(tf.random_normal([128, 10])),
            # out
            'out': tf.Variable(tf.random_normal([10, 1]))
        }
        
        biases = {
            # conv
            'bc1': tf.Variable(tf.random_normal([24])),
            'bc2': tf.Variable(tf.random_normal([48])),
            'bc3': tf.Variable(tf.random_normal([64])),
            # full conn
            'bd1': tf.Variable(tf.random_normal([1024])),
            'bd2': tf.Variable(tf.random_normal([128])),
            'bd3': tf.Variable(tf.random_normal([10])),
            # out
            'out': tf.Variable(tf.random_normal([1]))
        }

        self.pred = self.conv_net(self.x, weights, biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.square(self.pred - self.y))
        
    def run(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * self.batch_size < self.training_iters:
                batch_x, batch_y = self.data.read(self.batch_size)
                sess.run(optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})
                if step % self.display_step == 0:
                    loss = sess.run(self.cost, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})
                    print("Iter " + str(step * self.batch_size) + ", Minibatch Loss= " + "{:.16f}".format(loss))
                step += 1

    def predict(self, image):
        pass


if __name__ == '__main__':
    #import pdb;pdb.set_trace()
    model = Trainer('665')
    model.build_net()
    model.run()
