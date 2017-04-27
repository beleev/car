import argparse
import os
import sys
import h5py

import numpy
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 roc=True):
        image_num = len(images)
        assert image_num == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (image_num, labels.shape))
        self._num_examples = image_num

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)

        #if roc:
        #    images = images.roc(images.shape[0], images.shape[1] * images.shape[2])

        # Convert from [0, 255] -> [0.0, 1.0].
        for i in range(image_num):
            images[i] = numpy.multiply(images[i].astype(numpy.float32), 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   roc=True,
                   validation_percent=0.2):
    
    labelfile = train_dir + '/attr.h5'
    with h5py.File(labelfile, 'r') as f:
        labels = f['attrs'][:,3]

    imagefile = train_dir + '/image.h5'
    with h5py.File(imagefile, 'r') as f:
        images = [numpy.array(i) for i in f.values()]

    image_num = len(images)
    validation_size = int(image_num * validation_percent)
    if not 0 <= validation_size <= image_num:
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(image_num, validation_size))

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    train = DataSet(train_images, train_labels, roc=roc)
    validation = DataSet(validation_images,
                         validation_labels,
                         roc=roc)
    return base.Datasets(train=train, validation=validation, test=None)


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    image_num = len(images)
    if image_num != num_examples or num_examples == 0:
        raise ValueError('Images size %d does not match label size %d.' %
                                         (image_num, num_examples))
    rows = images[0].shape[0]
    cols = images[0].shape[1]
    depth = images[0].shape[2]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _float_feature(labels[index]),
                'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    # Get the data.
    data_sets = read_data_sets(FLAGS.directory,
                               roc=False,
                               validation_percent=FLAGS.validation_percent)

    # Convert to Examples and write the result to TFRecords.
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/root/car/data/665',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--validation_percent',
        type=int,
        default=0.2,
        help="""\
        Number of percent examples to separate from the training data 
        for the validation set.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
