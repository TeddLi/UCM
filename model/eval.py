import tensorflow as tf
import numpy as np
import os
import pdb
import sys
import time
import datetime
import operator
from collections import defaultdict
from model import data_helpers


# Files
tf.flags.DEFINE_string("test_file", "", "path to test file")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("char_vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("output_file", "", "prediction output file")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_utter_len", 50, "max utterance length")
tf.flags.DEFINE_integer("max_utter_num", 10, "max utterance number")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr in sorted(FLAGS.__flags.keys()):
    print("{}={}".format(attr.upper(), getattr(FLAGS, attr)))
print("")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))
charVocab = data_helpers.load_char_vocab(FLAGS.char_vocab_file)
print('charVocab size: {}'.format(len(charVocab)))

test_dataset = data_helpers.load_dataset(FLAGS.test_file, vocab, FLAGS.max_utter_num, FLAGS.max_utter_len)
print('test_dataset: {}'.format(len(test_dataset)))

target_loss_weight=[1.0,1.0]

print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
# pdb.set_trace()
graph = tf.Graph()
file = open(FLAGS.output_file, 'w')
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)



        # Get the placeholders from the graph by name
        utterances = graph.get_operation_by_name("utterances").outputs[0]
        utterances_len = graph.get_operation_by_name("utterances_len").outputs[0]
        utterances_num = graph.get_operation_by_name("utterances_num").outputs[0]
        u_char_feature = graph.get_operation_by_name("utterances_char").outputs[0]
        u_char_len     = graph.get_operation_by_name("utterances_char_len").outputs[0]
        # dialogue_label = graph.get_operation_by_name("dialogue_lable").outputs[0]
        target = graph.get_operation_by_name("target").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        probs = graph.get_operation_by_name("prediction_layer/probs").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        prediction_utterance = graph.get_operation_by_name("accuracy/prediction_utterance").outputs[0]

        results = defaultdict(list)
        num_test = 0
        num_correct = 0
        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, charVocab, FLAGS.max_word_length, shuffle=False)
        for test_batch in test_batches:
            x_utterances, x_utterances_len, x_utterances_num, x_utterances_char, x_utterances_char_len, x_target, x_id, x_target_weight, dialogue_lable = test_batch
            feed_dict = {
                utterances: x_utterances,
                utterances_len: x_utterances_len,
                utterances_num: x_utterances_num,
                dropout_keep_prob: 1.0,
                target: x_target,
                u_char_feature: x_utterances_char,
                u_char_len: x_utterances_char_len,
                # dialogue_label: dialogue_lable,
            }


            predicted_accuracy, predicted_probs, prediction_utter = sess.run([accuracy, probs, prediction_utterance], feed_dict)
            num_test += len(predicted_probs)
            print('num_test_sample={}'.format(num_test))
            # num_correct += len(predicted_probs) * predicted_accuracy
            # # pdb.set_trace()
            # predicted_target = np.argmax(predicted_probs, axis=2)

            predicted_target = np.argmax(predicted_probs, axis=2)  # [batch_size, max_utter_num]
            for i in range(len(predicted_probs)):
                i_utterances_num = x_utterances_num[i]
                i_predicted_target = predicted_target[i][:i_utterances_num]
                i_target = x_target[i][:i_utterances_num]
                if np.sum((i_predicted_target == i_target).astype(int)) == x_utterances_num[i]:
                    num_correct += 1

            for i in range(len(x_id)):
                x_id_ = x_id[i]
                i_utterances_num = x_utterances_num[i]
                for j in range(i_utterances_num):
                    i_predicted_target = predicted_target[i][j]
                    i_target = x_target[i][j]
                    file.write(str(x_id_))
                    file.write('\t')
                    file.write(str(i_utterances_num))
                    file.write('\t')
                    file.write(str(i_predicted_target))
                    file.write('\t')
                    file.write(str(i_target))
                    file.write('\n')

        acc = num_correct / num_test
        print('num_test_samples: {}  accuracy: {}'.format(num_test, acc))
            


