import argparse
import os
import sys
import h5py

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_data_sets(train_dir,
                   id_num,
                   roc=True,
                   validation_size=5000):
    pass
    
    


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    times = data_set.times
    curvs = data_set.curvs
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _float64_feature(curvs[index]),
                'image': image}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    # Get the data.
    data_sets = read_data_sets(FLAGS.directory,
        dtype=tf.uint8,
        reshape=False,
        validation_size=FLAGS.validation_size)

    # Convert to Examples and write the result to TFRecords.
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/tmp/data',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--validation_size',
        type=int,
        default=5000,
        help="""\
        Number of examples to separate from the training data for the validation
        set.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
