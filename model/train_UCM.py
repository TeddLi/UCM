import tensorflow as tf
import numpy as np
import os
import time
import datetime
import operator
import sys
import random
from tqdm import tqdm

from collections import defaultdict
from model import data_helpers
from model.model_UCM import UCM

# Files
tf.flags.DEFINE_string("train_file", "../data/Task3/3_Advising_dstc_valid.txt", "path to train file")
tf.flags.DEFINE_string("valid_file", "../data/Task3/task3/3_Advising_dstc_valid.txt", "path to valid file")
tf.flags.DEFINE_string("test_file", "../data/Task3/3_Advising_dstc_test.txt", "vocabulary file")
tf.flags.DEFINE_string("vocab_file", "../data/Task3/vocab.txt", "vocabulary file")
tf.flags.DEFINE_string("char_vocab_file",  "../data/Task3/char_vocab.txt", "path to char vocab file")
tf.flags.DEFINE_string("embedded_vector_file", "../data/Task3/glove_840B_300d_vec_plus_word2vec_100.txt", "pre-trained embedded word vector")
tf.flags.DEFINE_string('model_name',help=('model_name'), default = 'attr11')


# Model Hyperparameters
tf.flags.DEFINE_integer("max_utter_len", 10, "max utterance length")
tf.flags.DEFINE_integer("max_utter_num", 50, "max utterance number")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")
tf.flags.DEFINE_integer("num_layer", 3, "max response length")
tf.flags.DEFINE_integer("embedding_dim", 200, "dimensionality of word embedding")
tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")
tf.flags.DEFINE_integer("decay_step", 5000, "max utterance length")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "batch size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability (default: 1.0)")
tf.flags.DEFINE_integer("num_epochs", 1000000, "number of training epochs (default: 1000000)")
tf.flags.DEFINE_integer("evaluate_every", 1, "evaluate model on valid dataset after this many steps (default: 1000)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
# tf.app.flags.FLAGS.flag_values_dict()
for attr in sorted(FLAGS.__flags.keys()):
    print("{}={}".format(attr.upper(), getattr(FLAGS, attr)))
print("")

# Load data
print("Loading data...")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))
charVocab = data_helpers.load_char_vocab(FLAGS.char_vocab_file)
print('charVocab size: {}'.format(len(charVocab)))

train_dataset = data_helpers.load_dataset(FLAGS.train_file, vocab, FLAGS.max_utter_num, FLAGS.max_utter_len)
print('train_dataset: {}'.format(len(train_dataset)))
valid_dataset = data_helpers.load_dataset(FLAGS.valid_file, vocab, FLAGS.max_utter_num, FLAGS.max_utter_len)
print('valid_dataset: {}'.format(len(valid_dataset)))
test_dataset = data_helpers.load_dataset(FLAGS.test_file, vocab, FLAGS.max_utter_num, FLAGS.max_utter_len)
print('valid_dataset: {}'.format(len(valid_dataset)))


target_loss_weight=[1.0,1.0]


from tensorflow import set_random_seed
set_random_seed(25)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=session_conf)
    #random.seed(150)
    with sess.as_default():
        model = UCM(
            max_utter_len=FLAGS.max_utter_len,
            max_utter_num=FLAGS.max_utter_num,
            num_layer=FLAGS.num_layer,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            vocab=vocab,
            rnn_size=FLAGS.rnn_size,
            maxWordLength=FLAGS.max_word_length,
            charVocab=charVocab,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                   FLAGS.decay_step, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars1 = optimizer.compute_gradients(model.conv_loss)
        grads_and_vars2 = optimizer.compute_gradients(model.sentence_loss + model.conv_loss)
        train_op1 = optimizer.apply_gradients(grads_and_vars1, global_step=global_step)
        train_op2 = optimizer.apply_gradients(grads_and_vars2, global_step=global_step)
        train_op = train_op1

        # Keep track of gradient values and sparsity (optional)
        """
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """

        # Output directory for models and summaries
        timestamp = time.asctime( time.localtime(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_name + timestamp))
        print("Writing to {}\n".format(out_dir))


        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_utterances, x_utterances_len, x_utterances_num, x_utterances_char, x_utterances_char_len, x_target, x_id, x_target_weight, distance_mask, dialogue_lable):
            """
            A single training step
            """
            feed_dict = {
              model.utterances: x_utterances,
              model.utterances_len: x_utterances_len,
              model.utterances_num: x_utterances_num,
              model.target: x_target,
              model.dis_mask: distance_mask,
              model.target_loss_weight: x_target_weight,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob,
              model.u_charVec: x_utterances_char,
              model.u_charLen: x_utterances_char_len,
              model.dialogue_lable:dialogue_lable,
            }

            _, step, loss, conv_loss,accuracy, predicted_prob, debug = sess.run(
                [train_op, global_step, model.sentence_loss, model.conv_loss, model.accuracy, model.probs, model.debug_package],
                feed_dict)
            # avg, max, c_logits, mask_utter_num1, acc = debug

            if step%100 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)


        def check_step(dataset, shuffle, is_test = False, path = None):
            results = defaultdict(list)
            num_test = 0
            num_correct = 0.0
            conv_correct = 0.0
            if is_test:
                file = open(path, 'w')
            valid_batches = data_helpers.batch_iter(dataset, FLAGS.batch_size, 1, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, charVocab, FLAGS.max_word_length, shuffle=shuffle)
            for valid_batch in valid_batches:
                x_utterances, x_utterances_len, x_utterances_num, x_utterances_char, x_utterances_char_len, x_target, x_id, x_target_weight, dialogue_lable = valid_batch
                feed_dict = {
                  model.utterances: x_utterances,
                  model.utterances_len: x_utterances_len,
                  model.utterances_num: x_utterances_num,
                  model.target: x_target,
                  model.dis_mask: distance_mask,
                  model.target_loss_weight: x_target_weight,
                  model.dropout_keep_prob: 1.0,
                  model.u_charVec: x_utterances_char,
                  model.u_charLen: x_utterances_char_len,
                  model.dialogue_lable: dialogue_lable,
                }

                batch_accuracy, predicted_prob, conv_acc = sess.run([model.accuracy, model.probs, model.conv_acc], feed_dict)
                num_test += len(predicted_prob)

                if num_test %100000==0:
                    print(num_test)



                # method 1
                conv_correct += len(dialogue_lable) * conv_acc

                # method 2
                predicted_target = np.argmax(predicted_prob, axis=2)   # [batch_size, max_utter_num]
                for i in range(len(predicted_prob)):
                    i_utterances_num = x_utterances_num[i]
                    i_predicted_target = predicted_target[i][:i_utterances_num]
                    i_target = x_target[i][:i_utterances_num]
                    if np.sum((i_predicted_target == i_target).astype(int)) == x_utterances_num[i]:
                        num_correct += 1
                if is_test:
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



            # calculate Accuracy
            acc = num_correct / num_test
            cov_acc = conv_correct/num_test
            print('num_test_samples: {}  accuracy: {} \n'.format(num_test, acc))
            print('conversation accuracy: {} \n'.format(cov_acc))
            if is_test:
                file.close()
            return acc


        EPOCH = 0
        best_acc = 0.0
        batches = data_helpers.batch_iter(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, charVocab, FLAGS.max_word_length, shuffle=True)
        for batch in batches:
            x_utterances, x_utterances_len, x_utterances_num, x_utterances_char, x_utterances_char_len, x_target, x_id, x_target_weight, distance_mask, dialogue_lable = batch
            train_step(x_utterances, x_utterances_len, x_utterances_num, x_utterances_char, x_utterances_char_len, x_target, x_id, x_target_weight, distance_mask, dialogue_lable)
            current_step = tf.train.global_step(sess, global_step)
            if current_step == 10000:
                train_op = train_op2
                print('change to train_op2')
            if current_step % FLAGS.evaluate_every == 0:
                EPOCH += 1
                print("\nEPOCH: {}".format(EPOCH))
                print("\ncurrent_step: {}".format(current_step))
                print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                
                if EPOCH % 10 ==0:
                    print("Evaluation on Train set:")
                    train_acc = check_step(train_dataset, shuffle=False)
                print("Evaluation on Valid set:")
                valid_acc = check_step(valid_dataset, shuffle=False)
                if valid_acc > best_acc:
                    print("==========================")
                    print("Best Valid accuracy update")
                    print("==========================")
                    best_acc = valid_acc
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    print('start Testing')
                    test_acc = check_step(test_dataset, shuffle=False, is_test=True, path = os.path.join(out_dir,str(EPOCH)))
                    print('Test result = {}\n'.format(test_acc))
                if EPOCH>FLAGS.num_epochs:
                    break


