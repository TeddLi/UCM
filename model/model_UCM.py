import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS



def LDAMLoss(logit, labels, m_list, weight=None, s=30):
    
    num_label = len(m_list)
    index = tf.one_hot(labels, num_label, dtype=tf.int32)
    index_float = tf.cast(index, dtype=tf.float32)
    m_list = 1.0 / np.sqrt(np.sqrt(m_list))
    m_list = (m_list * (0.5 / np.max(m_list))).astype('float32')
    batch_m = tf.matmul(index_float, tf.transpose(m_list[None,:]))
    batch_m = tf.reshape(batch_m, [-1, 1])
    x_m = logit - batch_m
    y = tf.constant([0])
    index_bool = tf.greater(index, y)
    output = tf.where(index_bool, x_m, logit)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s*output, labels = labels)
    return loss





def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")
    # return tf.get_variable(initializer=initializer, name="word_embedding")


def get_char_embedding(charVocab):
    print("get_char_embedding")
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0
    return tf.constant(embeddings, name="word_char_embedding")
    # return tf.get_variable(initializer=embeddings, name="word_char_embedding")


def load_embed_vectors(fname, dim):
    # vectors = { 'the': [0.2911, 0.3288, 0.2002,...], ... }
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim + 1)]
        vectors[items[0]] = vec
    return vectors


def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        # else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim)
    return embeddings


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                  inputs=inputs,
                                                                  sequence_length=input_seq_len,
                                                                  dtype=tf.float32)
        return rnn_outputs, rnn_states


def multi_lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, num_layer, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        multi_outputs = []
        multi_states = []
        cur_inputs = inputs
        for i_layer in range(num_layer):
            rnn_outputs, rnn_states = lstm_layer(cur_inputs, input_seq_len, rnn_size, dropout_keep_prob,
                                                 scope + str(i_layer), scope_reuse)
            rnn_outputs = tf.concat(values=rnn_outputs, axis=2)
            multi_outputs.append(rnn_outputs)
            multi_states.append(rnn_states)
            cur_inputs = rnn_outputs

        # multi_layer_aggregation
        ml_weights = tf.nn.softmax(
            tf.get_variable("ml_scores", [num_layer, ], initializer=tf.constant_initializer(0.0)))

        multi_outputs = tf.stack(multi_outputs, axis=-1)  # [batch_size, max_len, 2*rnn_size(400), num_layer]
        max_len = multi_outputs.get_shape()[1].value
        dim = multi_outputs.get_shape()[2].value
        flattened_multi_outputs = tf.reshape(multi_outputs,
                                             [-1, num_layer])  # [batch_size * max_len * 2*rnn_size(400), num_layer]
        aggregated_ml_outputs = tf.matmul(flattened_multi_outputs,
                                          tf.expand_dims(ml_weights, 1))  # [batch_size * max_len * 2*rnn_size(400), 1]
        aggregated_ml_outputs = tf.reshape(aggregated_ml_outputs,
                                           [-1, max_len, dim])  # [batch_size , max_len , 2*rnn_size(400)]

        return aggregated_ml_outputs


def cnn_layer(inputs, filter_sizes, num_filters, scope=None, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[2].value

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(i)):
                w = tf.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(inputs, w, stride=1,
                                padding="VALID")  # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b))  # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
            outputs.append(pooled)
    return tf.concat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)]


class UCM(object):
    def __init__(
            self, max_utter_len, max_utter_num,  vocab, rnn_size, maxWordLength,
            charVocab, l2_reg_lambda=0.0):
        self.utterances = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances")
        self.utterances_len = tf.placeholder(tf.int32, [None, max_utter_num], name="utterances_len")
        self.utterances_num = tf.placeholder(tf.int32, [None], name="utterances_num")

        self.u_charVec = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len, maxWordLength],
                                        name="utterances_char")
        self.u_charLen = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances_char_len")
        self.dialogue_lable =  tf.placeholder(tf.int64, [None], name="dialogue_lable")
        self.target = tf.placeholder(tf.int64, [None, max_utter_num], name="target")
        self.target_loss_weight = tf.placeholder(tf.float32, [None], name="target_weight")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(1.0)

        # =============================== Embedding layer ===============================
        # 1. word embedding
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W,
                                                         self.utterances)  # [batch_size, max_utter_num, max_utter_len,  word_dim]
            print("original utterances_embedded: {}".format(utterances_embedded.get_shape()))

        with tf.name_scope('char_embedding'):
            char_W = get_char_embedding(charVocab)
            utterances_char_embedded = tf.nn.embedding_lookup(char_W,
                                                              self.u_charVec)  # [batch_size, max_utter_num, max_utter_len,  maxWordLength, char_dim]
            print("utterances_char_embedded: {}".format(utterances_char_embedded.get_shape()))

        charRNN_size = 40
        charRNN_name = "char_RNN"
        char_dim = utterances_char_embedded.get_shape()[-1].value
        utterances_char_embedded = tf.reshape(utterances_char_embedded, [-1, maxWordLength,
                                                                         char_dim])  # [batch_size*max_utter_num*max_utter_len, maxWordLength, char_dim]

        # 3. char CNN
        utterances_cnn_char_emb = cnn_layer(utterances_char_embedded, filter_sizes=[3, 4, 5], num_filters=50,
                                            scope="CNN_char_emb",
                                            scope_reuse=False)  # [batch_size*max_utter_num*max_utter_len,   emb]
        cnn_char_dim = utterances_cnn_char_emb.get_shape()[1].value
        utterances_cnn_char_emb = tf.reshape(utterances_cnn_char_emb, [-1, max_utter_num, max_utter_len,
                                                                       cnn_char_dim])  # [batch_size, max_utter_num, max_utter_len, emb]

        utterances_embedded = tf.concat(axis=-1, values=[utterances_embedded,
                                                         utterances_cnn_char_emb])  # [batch_size, max_utter_num, max_utter_len, emb]
        utterances_embedded = tf.nn.dropout(utterances_embedded, keep_prob=self.dropout_keep_prob)
        print("utterances_embedded: {}".format(utterances_embedded.get_shape()))

        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "bidirectional_rnn"
            emb_dim = utterances_embedded.get_shape()[-1].value
            flattened_utterances_embedded = tf.reshape(utterances_embedded, [-1, max_utter_len,
                                                                             emb_dim])  # [batch_size*max_utter_num, max_utter_len, emb]
            flattened_utterances_len = tf.reshape(self.utterances_len, [-1])  # [batch_size*max_utter_num, ]
            # 1. single_lstm_layer
            u_rnn_output, u_rnn_states = lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size,
                                                    self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
            utterances_output = tf.concat(axis=2,
                                          values=u_rnn_output)  # [batch_size*max_utter_num,  max_utter_len, rnn_size*2]
            print("establish single_lstm_layer")
            # 2. multi_lstm_layer
            # utterances_output = multi_lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=False)
            output_dim = utterances_output.get_shape()[-1].value

        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            final_utterances_max = tf.reduce_max(utterances_output, axis=1)
            final_utterances_state = tf.concat(axis=1, values=[u_rnn_states[0].h, u_rnn_states[1].h])
            final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])
            print("establish aggregation of max pooling and last-state pooling")
            # concat_dim = final_utterances.get_shape()[-1].value
            rnn_scope_aggre = "bidirectional_rnn_aggregation"
            final_utterances = tf.reshape(final_utterances, [-1, max_utter_num,
                                                             output_dim * 2])  # [batch_size, max_utter_num, 4*rnn_size]
            utterances_output, utterances_state = lstm_layer(final_utterances, self.utterances_num, rnn_size,
                                                             self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=False)
            utterances_output = tf.concat(axis=2, values=utterances_output)  # [batch_size, max_utter_num, 2*rnn_size]
            print("utterances_output: {}".format(utterances_output.get_shape()))

        # =============================== Context attention layer ===============================
        with tf.variable_scope("Context_attention_layer") as vs:
            # self_attention 1
            W1 = tf.get_variable("W1", [output_dim, output_dim*2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            B1 = tf.get_variable("B1", [output_dim*2], initializer=tf.zeros_initializer())

            W2 = tf.get_variable("W2", [output_dim*2, output_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            B2 = tf.get_variable("B2", [output_dim], initializer=tf.zeros_initializer())

            attr_input = tf.reshape(utterances_output, [-1, output_dim])
            Att1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(attr_input, W1), B1))
            Att2 = tf.nn.softmax(tf.nn.bias_add(tf.matmul(Att1, W2), B2), axis=-1) # [batch_size* max_utter_num, 2*rnn_size]
            Att2 = tf.reshape(Att2, [-1, max_utter_num, output_dim])
            final_utterances_output = Att2 * utterances_output
            final_utterances_output = tf.concat([utterances_output, final_utterances_output], axis=2)


            print("final_utterances_output: {}".format(final_utterances_output.get_shape()))

        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            hidden_input_size = final_utterances_output.get_shape()[-1].value
            joined_feature = tf.reshape(final_utterances_output, [-1, hidden_input_size])
            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            # regularizer = None
            # dropout On MLP
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                         activation_fn=tf.nn.relu,
                                                         reuse=False,
                                                         trainable=True,
                                                         scope="projected_layer")  # [batch_size*max_utter_num, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)

            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[3]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 3], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(full_out, s_w) + bias  # [batch_size*max_utter_num, 3]
            print("logits: {}".format(logits.get_shape()))

            # logits = tf.squeeze(logits, [1])               # [batch_size, ]
            # self.probs = tf.sigmoid(logits, name="prob")   # [batch_size, ]
            flatten_probs = tf.nn.softmax(logits, name="flatten_probs")  # [batch_size*max_utter_num, n_class(3)]
            self.probs = tf.reshape(flatten_probs, [-1, max_utter_num, 3],
                                    name="probs")  # [batch_size, max_utter_num, n_class(3)]

            flatten_target = tf.reshape(self.target, [-1, ])  # [batch_size*max_utter_num, ]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flatten_target)

            mask_utter_num1 = tf.sequence_mask(self.utterances_num, max_utter_num,
                                              dtype=tf.float32)  # [batch_size, max_utter_num]
            mask_utter_num = tf.reshape(mask_utter_num1, [-1, ])  # [batch_size*max_utter_num, ]
            losses = tf.multiply(losses, mask_utter_num)
            losses = tf.multiply(losses, self.target_loss_weight)
            all_loss = tf.reduce_sum(losses)
            all_valid = tf.reduce_sum(mask_utter_num)

            self.sentence_loss = all_loss/all_valid + l2_reg_lambda * l2_loss + sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("Conversation_loss"):
            self.conv_prob = 1 - self.probs
            self.conv_prob1 = 1 - tf.reduce_prod(self.conv_prob, axis=1)
            self.probs1 = self.probs * mask_utter_num1[:, :, None]
            avg = tf.reduce_sum(self.probs1, axis=1) / tf.reduce_sum(mask_utter_num1, axis=1)[:, None]
            max = tf.reduce_max(self.probs1, axis=1)
            feature = tf.concat([avg, max], axis=-1)
            c_bias = tf.Variable(tf.constant(0.1, shape=[2]), name="c_bias")
            c_w = tf.get_variable("c_w", shape=[6, 2], initializer=tf.contrib.layers.xavier_initializer())
            c_logits = tf.matmul(feature, c_w) + c_bias  # [batch_size*max_utter_num, 3]
            self.conv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=c_logits, labels=self.dialogue_lable)
            conv_probs = tf.argmax(tf.nn.softmax(c_logits), 1)

            self.conv_acc = tf.reduce_mean(tf.cast(tf.equal(conv_probs, self.dialogue_lable), dtype=tf.float32))
            self.debug_package = [avg, max, c_logits, conv_probs, self.conv_acc]



        with tf.name_scope("accuracy"):
            flatten_prediction_utterance = tf.argmax(flatten_probs, 1)  # [batch_size*max_utter_num, ]
            self.prediction_utterance = tf.reshape(flatten_prediction_utterance, [-1, max_utter_num],
                                                   name="prediction_utterance")  # [batch_size, max_utter_num]

            flatten_prediction_utterance_correct = tf.cast(tf.equal(flatten_prediction_utterance, flatten_target),
                                                           "float")  # [batch_size*max_utter_num, ]
            flatten_prediction_utterance_correct_mask = tf.multiply(flatten_prediction_utterance_correct,
                                                                    mask_utter_num)
            prediction_utterance_correct_mask = tf.reshape(flatten_prediction_utterance_correct_mask,
                                                           [-1, max_utter_num])  # [batch_size, max_utter_num]
            prediction_dialogue = tf.reduce_sum(prediction_utterance_correct_mask, 1)  # [batch_size, ]
            self.prediction_dialogue_correct = tf.cast(
                tf.equal(prediction_dialogue, tf.cast(self.utterances_num, "float")), "float",
                name="prediction_dialogue_correct")  # [batch_size, ]
            self.accuracy = tf.reduce_mean(self.prediction_dialogue_correct, name="accuracy")


