from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import curve_net_gen_images

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')
# Global constants describing the data set.
tf.app.flags.DEFINE_integer('image_size', 96, 'Image size.')
tf.app.flags.DEFINE_integer('image_channels', 1, 'Image channels.')
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 512, 'the size of the training set.')
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_eval', 64, 'the size of the eval set.')
CLASS_NAMES = ["Para1","Para2","Para3","hyper1","hyper2","hyper3"] # The names of classes

# Structure of the ground truth
tf.app.flags.DEFINE_integer('num_box_confidence', 1, 'the chars of box confidence in a box info.')
tf.app.flags.DEFINE_integer('num_box_cood', 4, 'the chars of box coodination (middle point of (x,y) and width and hight in a box info.')
tf.app.flags.DEFINE_integer('num_classes', 6, 'the chars of classes in a box info')
tf.app.flags.DEFINE_integer('num_boxes', 5, 'the number of boxes in a ground truth')
tf.app.flags.DEFINE_integer('num_outputs', 55, 'the number of output chars in a ground truth') #(FLAGS.num_box_confidence + FLAGS.num_box_cood + FLAGS.num_classes) * FLAGS.num_boxes

# Structure of the data folder 
tf.app.flags.DEFINE_string('train_data_dir', 'data_batch', 'the folder of the training data')
tf.app.flags.DEFINE_string('eval_data_dir', 'eval_batch', 'the folder of the eval data')
DATA_FILE_NAME = "data.csv"


def read_curve_net(filename_queue, file_dir):
    """Reads and parses examples from curveNet data files.

    Args:
        filename_queue: A queue of strings with the filenames to read from.
        file_dir: image files forlder
    Returns:
        Y_labels: an float32 Tensor with a number of labels 
                (FLAGS.num_outputs - FLAGS.num_box_confidence + FLAGS.num_box_cood + FLAGS.num_classes) * FLAGS.num_boxes in a single example.
        image: a [FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels] Tensor with the image data
        img_name: an string tensor with the file name of image
    """
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    record_defaults = [[""]] + [[1.0]] * FLAGS.num_outputs
    components = tf.decode_csv(value, record_defaults=record_defaults)
    img_name = components[0]
    Y_labels = tf.convert_to_tensor(components[1:], dtype=tf.float32)
    image_path = file_dir + '/' + img_name
    img_contents = tf.read_file(image_path)
    img = tf.image.decode_png(img_contents, channels=FLAGS.image_channels)
    
    return img, Y_labels, img_name

def _generate_image_and_labels_batch(image, labels, img_name, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and Y_labels.
    Args:
        image: 3-D Tensor of [FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels] of type.float32.
        labels: 2-D Tensor of [1, FLAGS.num_outputs]
        img_name: 1-D Tensor of [1] of type.string
        min_queue_examples: int32, minimum number of samples to retain
                            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels] size.
        labels: Y_labels. 2D tensor of [batch_size, FLAGS.num_outputs] size.
        img_names: Image names. 1D tensor of [batch_size] size
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels_batch from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, labels_batch, img_names = tf.train.shuffle_batch(
                                            [image, labels, img_name],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batch_size,
                                            min_after_dequeue=min_queue_examples)
    else:
        images, labels_batch, img_names = tf.train.batch(
                                            [image, labels, img_name],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, labels_batch, img_names

def inputs(eval_data, data_dir, batch_size):
    """Construct input for CurveNet evaluation using the Reader ops.
    
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CurveNET data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels] size.
        labels: Labels. 2D tensor of [batch_size, FLAGS.num_outputs] size.
        img_name: Image names. 1D tensor of [batch_size] size
    """
    if not eval_data:
        file_dir = os.path.join(data_dir, FLAGS.train_data_dir)
        filenames = [os.path.join(file_dir, DATA_FILE_NAME)]
        num_examples_per_epoch = FLAGS.num_examples_per_epoch_for_train
    else:
        file_dir = os.path.join(data_dir, FLAGS.eval_data_dir)
        filenames = [os.path.join(file_dir, DATA_FILE_NAME)]
        num_examples_per_epoch = FLAGS.num_examples_per_epoch_for_eval
    
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
            
    with tf.name_scope('input'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)
        
        img, labels, img_name = read_curve_net(filename_queue, file_dir)
        reshaped_image = tf.cast(img, tf.float32)
        
        height = FLAGS.image_size
        width = FLAGS.image_size
        
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(reshaped_image)
        
        # Set the shapes of tensors.
        float_image = tf.reshape(float_image, [height, width, FLAGS.image_channels])
        labels = tf.reshape(labels, [FLAGS.num_outputs])
        img_name = tf.reshape(img_name, [1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_labels_batch(float_image, labels, img_name,
                                              min_queue_examples, batch_size,
                                              shuffle=(not eval_data))

def generate_images(data_dir):
    """Generate training and test images to simulate the real image.
    
    Args:
        data_dir: Path to the CurveNET data directory.
    """
    
    print ("Start to generate %i training images and %i test images with the image size %i x %i" % (FLAGS.num_examples_per_epoch_for_train, FLAGS.num_examples_per_epoch_for_eval, FLAGS.image_size, FLAGS.image_size))
    if tf.gfile.Exists(data_dir):
        tf.gfile.DeleteRecursively(data_dir)
        
    # Generate training data
    training_data_dir = os.path.join(data_dir, FLAGS.train_data_dir)
    tf.gfile.MakeDirs(training_data_dir)
    curve_net_gen_images.gen_image(training_data_dir, 
                                      img_num=FLAGS.num_examples_per_epoch_for_train,
                                      max_line_num=FLAGS.num_boxes,
                                      max_func_num=FLAGS.num_classes,
                                      image_size=FLAGS.image_size)

    # Generate eval data
    eval_data_dir = os.path.join(data_dir, FLAGS.eval_data_dir)
    tf.gfile.MakeDirs(eval_data_dir)
    curve_net_gen_images.gen_image(eval_data_dir, 
                                      img_num=FLAGS.num_examples_per_epoch_for_eval,
                                      max_line_num=FLAGS.num_boxes,
                                      max_func_num=FLAGS.num_classes,
                                      image_size=FLAGS.image_size)