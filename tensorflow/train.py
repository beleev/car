import argparse
import os
import sys

import tensorflow as tf

FLAGS = None

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
IMAGE_PIXELS = 320 * 320 * 3


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        })
  
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image.set_shape([IMAGE_PIXELS])
  
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
  
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = image - 0.5
  
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.float32)
  
    return image, label


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.directory,
                            TRAIN_FILE if train else VALIDATION_FILE)
  
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)
  
        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)
  
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)
  
    return images, sparse_labels


def run_training():
    images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input images and labels.
        images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs)
    
        # Build a Graph that computes predictions from the inference model.
        logits = mnist.inference(images,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)
    
        # Add to the Graph the loss calculation.
        loss = mnist.loss(logits, labels)
    
        # Add to the Graph operations that train the model.
        train_op = mnist.training(loss, FLAGS.learning_rate)
    
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    
        # Create a session for running operations in the Graph.
        sess = tf.Session()
    
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
    
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        try:
            step = 0
            while not coord.should_stop():
              start_time = time.time()
    
              # Run one step of the model.  The return values are
              # the activations from the `train_op` (which is
              # discarded) and the `loss` op.  To inspect the values
              # of your ops or variables, you may include them in
              # the list passed to sess.run() and the value tensors
              # will be returned in the tuple from the call.
              _, loss_value = sess.run([train_op, loss])
    
              duration = time.time() - start_time
    
              # Print an overview fairly often.
              if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                           duration))
              step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
    
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
    

def main(unused_argv):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/root/car/data/665',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)