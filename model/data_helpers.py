import numpy as np
import random
import json
import pdb
from nltk.tokenize import WordPunctTokenizer


def tokenize(text):
    return WordPunctTokenizer().tokenize(text)

def load_vocab(fname):
    '''
    vocab = {"<PAD>": 0, ...}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for term_id, line in enumerate(f):
            line = line.strip()
            fields = line.split('\t')
            # term_id = int(fields[1])
            vocab[fields[0]] = term_id
    return vocab

def load_char_vocab(fname):
    '''
    charVocab = {"U": 0, "!": 1, ...}
    '''
    charVocab={}
    with open(fname, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            char_id = int(fields[0])
            ch = fields[1]
            charVocab[ch] = char_id
    return charVocab

def to_vec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["UNKNOWN"])

    return length, np.array(vec)



    # 2. create the label
    label = [0] * len(utterances)
    for success_label in dialog["success-labels"]:
        if success_label["label"] == "Accept":
              label[success_label["position"]] = 2
        elif success_label["label"] == "Reject":
              label[success_label["position"]] = 1

    return utterances_id, utterances, utterances_speaker, label


def load_dataset(fname, vocab, max_utter_num, max_utter_len):

    dataset=[]
    with open(fname, 'r') as f:
        us_id = []
        utterances = []
        utterances_speaker = []
        label = []
        for line in f:
            if line != '\n':
                temp = line.rstrip().split('\t')
                us_id.append(int(temp[0]))
                utterances.append(temp[1])
                utterances_speaker.append(temp[2])
                label.append(int(temp[3]))
            elif line =='\n':
                utterances = utterances[-max_utter_num:]   # select the last max_utter_num utterances
                label = label[-max_utter_num:]
                us_id = us_id[-max_utter_num:]
                utterances_speaker = utterances_speaker[-max_utter_num:]

                us_num = len(utterances)

                us_tokens = []
                us_vec = []
                us_len = []
                for utterance in utterances:
                    u_tokens = tokenize(utterance)[:max_utter_len]
                    # u_tokens = utterance.split(' ')[:max_utter_len]  # select the first max_utter_len tokens in every utterance
                    u_len, u_vec = to_vec(u_tokens, vocab, max_utter_len)
                    us_tokens.append(u_tokens)
                    us_vec.append(u_vec)
                    us_len.append(u_len)

                dataset.append((us_id, us_tokens, us_vec, us_len, us_num, label, utterances_speaker))
                us_id = []
                utterances = []
                utterances_speaker = []
                label = []

    return dataset



def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec


def charVec(tokens, charVocab, maxlen, maxWordLength):
    '''
    chars = np.array( (maxlen, maxWordLength) )    0 if not found in charVocab or None
    word_lengths = np.array( maxlen )              1 if None
    '''
    n = len(tokens)
    if n > maxlen:
        n = maxlen

    chars =  np.zeros((maxlen, maxWordLength), dtype=np.int32)
    word_lengths = np.ones(maxlen, dtype=np.int32)
    for i in range(n):
        token = tokens[i][:maxWordLength]
        word_lengths[i] = len(token)
        row = chars[i]
        for idx, ch in enumerate(token):
            if ch in charVocab:
                row[idx] = charVocab[ch]

    return chars, word_lengths


def batch_iter(data, batch_size, num_epochs, target_loss_weights, max_utter_len, max_utter_num, charVocab, max_word_length, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.Random(epoch).shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            x_id = []
            x_utterances = []
            x_utterances_len = []
            x_utterances_num = []
            x_utterances_char=[]
            x_utterances_char_len=[]
            targets = []
            target_weights_all=[]
            Dialogue_lable = []

            for rowIdx in range(start_index, end_index):
                us_id, us_tokens, us_vec, us_len, us_num, label, speaker = data[rowIdx]
                target_weights = []

                # normalize us_vec and us_len
                new_utters_vec = np.zeros((max_utter_num, max_utter_len), dtype='int32')
                new_utters_len = np.zeros((max_utter_num, ), dtype='int32')
                mask = np.zeros((max_utter_num, max_utter_num), dtype='float32')
                for i in range(len(us_len)):
                    new_utter_vec = normalize_vec(us_vec[i], max_utter_len)
                    new_utters_vec[i] = new_utter_vec
                    new_utters_len[i] = us_len[i]
                x_utterances.append(new_utters_vec)
                x_utterances_len.append(new_utters_len)

                x_utterances_num.append(us_num)

                new_label = np.zeros((max_utter_num, ), dtype='int32')
                conv_label = 0
                for i in range(len(label)):
                    new_label[i] = label[i]
                    if label[i] > 0:
                        target_weights.append(target_loss_weights[1])
                        conv_label = 1
                    else:
                        target_weights.append(target_loss_weights[0])
                targets.append(new_label)
                Dialogue_lable.append((conv_label))

                for  j in range(max_utter_num - len(target_weights)):
                    target_weights.append(target_loss_weights[0])
                target_weights_all.extend(target_weights)


                x_id.append(us_id[0])

                # normalize CharVec and CharLen
                uttersCharVec = np.zeros((max_utter_num, max_utter_len, max_word_length), dtype='int32')
                uttersCharLen = np.ones((max_utter_num, max_utter_len), dtype='int32')
                for i in range(len(us_len)):
                    utterCharVec, utterCharLen = charVec(us_tokens[i], charVocab, max_utter_len, max_word_length)
                    uttersCharVec[i] = utterCharVec
                    uttersCharLen[i] = utterCharLen
                x_utterances_char.append(uttersCharVec)
                x_utterances_char_len.append(uttersCharLen)


            # pdb.set_trace()
            yield np.array(x_utterances), np.array(x_utterances_len), np.array(x_utterances_num), \
                  np.array(x_utterances_char), np.array(x_utterances_char_len), \
                  np.array(targets), x_id , np.array(target_weights_all), np.array(Dialogue_lable)

