from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow as tf

import curve_net as cn
import curve_net_utils as cnu

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'tmp/curve_net_eval', 'Directory where to write event logs.')
tf.app.flags.DEFINE_string('eval_data', 'test', 'Either test or train_eval.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/curve_net_train', 'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, 'How often to run the eval.')
tf.app.flags.DEFINE_boolean('run_once', False, 'Whether to run eval only once.')

CLASS_NAMES = cn.CLASS_NAMES

NUM_VISUALIZED_IMAGES_PER_BATCH = 10 # The number of visuzlized images per batch after evaluation.
MAX_POINT_RANGE = 10


def eval_once(saver, summary_writer, labels, logits, img_names, summary_op):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        labels: true values with the shape of [num_examples, num_outputs]
        logits: predicted values with the shape of [num_examples, num_outputs]
        img_names: use to visualize predicted result
        summary_op: Summary op.
    """
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/curve_net_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
          print('No checkpoint file found')
          return
      
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            
            num_iter = int(math.ceil(FLAGS.num_examples_per_epoch_for_eval / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                y, y_pred, image_names = sess.run([labels, logits, img_names])
                
                accuracy = cnu.calculate_accuracy(y, y_pred, FLAGS.num_box_confidence, 
                                                                  FLAGS.num_box_cood, FLAGS.num_classes,
                                                                  FLAGS.box_confidence_threshold, FLAGS.iou_threshold)
                true_count += accuracy * FLAGS.batch_size
                cnu.visualize_predicted_result(
                        y_pred[:NUM_VISUALIZED_IMAGES_PER_BATCH], image_names[:NUM_VISUALIZED_IMAGES_PER_BATCH],
                        CLASS_NAMES, FLAGS.data_dir, FLAGS.eval_data_dir, FLAGS.image_size, MAX_POINT_RANGE, FLAGS.box_confidence_threshold)
                
                step += 1
            
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
    
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
            
def evaluate():
    """Eval CurveNET for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CurveNET.
        eval_data = (FLAGS.eval_data == 'test')
        images, labels, img_names = cn.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cn.inference(images)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(cn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        
        while True:
            eval_once(saver, summary_writer, labels, logits, img_names, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
            
def main(argv=None):  # pylint: disable=unused-argument
    cn.maybe_generate_images()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()