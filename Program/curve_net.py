from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

import curve_net_input as cni

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of images to process in a batch.')
tf.app.flags.DEFINE_string('data_dir', 'tmp/curve_net_data', 'Path to the ParticleNET data directory.')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Train the model using fp16.')
tf.app.flags.DEFINE_float('box_confidence_threshold', 0.8, 'The threshold of box confidence.')
tf.app.flags.DEFINE_float('iou_threshold', 0.8, 'The threshold of IOU.')

CLASS_NAMES = cni.CLASS_NAMES

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999    # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
BN_EPSILON = 1e-3  # Batch normalization epsilon

filters_num = [32,64]
neurons_num = [1024,512]


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))
    

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def inputs(eval_data):
    """Construct input for ParticleNET evaluation using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
        images: Grey scale Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        labels: labels. 2D tensor of [batch_size, NUM_OUTPUTS] size.
    
    Raises:
        ValueError: If no data_dir
    """

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    
    #data_dir = os.path.join(FLAGS.data_dir, 'curve-net-batches-bin')
    images, labels, img_names = cni.inputs(eval_data=eval_data,
                                                           data_dir=FLAGS.data_dir,
                                                           batch_size=FLAGS.batch_size)
    
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    
    return images, labels, img_names
    

def inference(images):
    """Build the ParticleNet model.
    Args:
        images: Images returned from inputs().
    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
   
    REG_LAMBDA = tf.constant(5e-4, name='REG_LAMBDA')  # Regularization lambda
    
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_on_cpu('weights', [5, 5, FLAGS.image_channels, filters_num[0]], tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        
        # batch_normalization
        batch_mean1, batch_var1 = tf.nn.moments(conv,[0,1,2], name='moments')
        scale1 = _variable_on_cpu('scales', [filters_num[0]], tf.ones_initializer())
        beta1 = _variable_on_cpu('betas', [filters_num[0]], tf.zeros_initializer())
        bn = tf.nn.batch_normalization(conv, batch_mean1, batch_var1, beta1, scale1, BN_EPSILON)
        
        conv1 = tf.nn.relu(bn, name=scope.name)
        _activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_on_cpu('weights', [5, 5, filters_num[0], filters_num[1]], tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        
        # batch_normalization
        batch_mean2, batch_var2 = tf.nn.moments(conv,[0,1,2], name='moments')
        scale2 = _variable_on_cpu('scales', [filters_num[1]], tf.ones_initializer())
        beta2 = _variable_on_cpu('betas', [filters_num[1]], tf.zeros_initializer())
        bn = tf.nn.batch_normalization(conv, batch_mean2, batch_var2, beta2, scale2, BN_EPSILON)
        
        conv2 = tf.nn.relu(bn, name=scope.name)
        _activation_summary(conv2)

        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')
        
    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_on_cpu('weights', [dim, neurons_num[0]], tf.contrib.layers.xavier_initializer())
        tf.add_to_collection('losses', REG_LAMBDA * tf.nn.l2_loss(weights))
        biases = _variable_on_cpu('biases', [neurons_num[0]], tf.contrib.layers.xavier_initializer())
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)
    
    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_on_cpu('weights', [neurons_num[0], neurons_num[1]], tf.contrib.layers.xavier_initializer())
        tf.add_to_collection('losses', REG_LAMBDA * tf.nn.l2_loss(weights))
        biases = _variable_on_cpu('biases', [neurons_num[1]], tf.contrib.layers.xavier_initializer())
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX + b)
    with tf.variable_scope('linear') as scope:
        weights = _variable_on_cpu('weights', [neurons_num[1], FLAGS.num_outputs], tf.contrib.layers.xavier_initializer())
        tf.add_to_collection('losses', REG_LAMBDA * tf.nn.l2_loss(weights))
        biases = _variable_on_cpu('biases', [FLAGS.num_outputs], tf.contrib.layers.xavier_initializer())
        linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(linear)
    
    return linear

def loss(logits, labels):
    """Add L2 Loss to all the trainable variables, and calculate all losses 
        which generate from box confidence, coodinations and curve types
    
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: labels from inputs(). 2-D tensor of shape [batch_size, NUM_OUTPUTS]
    Returns:
        Loss tensor of type float.
    """
    box_column_size = FLAGS.num_box_confidence + FLAGS.num_box_cood + FLAGS.num_classes
    reshape_size = tf.constant([-1, box_column_size])
    #box_confidence_threshold = tf.constant(FLAGS.box_confidence_threshold, name='box_confidence_threshold')
    
    labels_reshape = tf.reshape(labels, reshape_size)
    logits_reshape = tf.reshape(logits, reshape_size)
    
    loss = 0
    
    #. Calculate the loss of box_confidence by sigmoid cross entropy
    box_conf_begin = tf.constant([0,0])
    box_conf_size = tf.constant([-1,FLAGS.num_box_confidence])
    box_conf_labels = tf.slice(labels_reshape, box_conf_begin, box_conf_size)
    box_conf_logits = tf.slice(logits_reshape, box_conf_begin, box_conf_size)
    loss_X = tf.nn.sigmoid_cross_entropy_with_logits(labels=box_conf_labels, logits=box_conf_logits)
    #box_conf_logits = tf.clip_by_value(tf.sigmoid(box_conf_logits),1e-10,1.0)
    #loss_X = box_conf_labels * -tf.log(box_conf_logits) + (1-box_conf_labels) * -tf.log(1-box_conf_logits)
    loss += tf.reduce_sum(loss_X, name='box_confidence_loss')
    
    #. Calculate the loss of coodinations by MSE (if box confidence is 0, then don't count the loss)
    cood_begin = tf.constant([0,FLAGS.num_box_confidence])
    cood_size = tf.constant([-1,FLAGS.num_box_cood])
    cood_labels = tf.slice(labels_reshape, cood_begin, cood_size)
    cood_logits = tf.slice(logits_reshape, cood_begin, cood_size)
    # if box_conf_label is 0, then loss will become 0
    loss_X = tf.reduce_sum(tf.squared_difference(cood_labels, cood_logits), axis=1) * box_conf_labels 
    loss += (tf.reduce_sum(loss_X)/tf.reduce_sum(box_conf_labels))
    
    #. Calculate the loss of curve type by softmax cross entropy (if box confidence is 0, then don't count the loss)
    curve_type_begin = tf.constant([0,FLAGS.num_box_confidence+FLAGS.num_box_cood])
    curve_type_size = tf.constant([-1,FLAGS.num_classes])
    curve_type_labels = tf.slice(labels_reshape, curve_type_begin, curve_type_size)
    curve_type_logits = tf.clip_by_value(tf.nn.softmax(tf.slice(logits_reshape, curve_type_begin, curve_type_size)),1e-10,1.0)
    loss_X = curve_type_labels * -tf.log(curve_type_logits) * box_conf_labels
    loss += tf.reduce_sum(loss_X, name='curve_type_loss')
    
    tf.add_to_collection('losses', loss)
    
    # The total loss is defined as the cross entropy loss plus all of the L2 losses.
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
def _add_loss_summaries(total_loss):
    """Add summaries for losses in ParticleNET model.
    
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """Train ParticleNET model.
    
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps processed.
    Returns:
        train_op: op for training.
    """
    
    # Variables that affect learning rate.
    num_batches_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op
    
def maybe_generate_images():
    """Generate training and eval images if there is no data existing"""
    dest_dir = FLAGS.data_dir
    if not os.path.exists(dest_dir):
        print ("The data does not exist, start to generate the images")
        os.makedirs(dest_dir)
        cni.generate_images(dest_dir)
        print ("The images generated")