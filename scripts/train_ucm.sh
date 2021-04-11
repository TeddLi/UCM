cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data

train_file=$DATA_DIR/Advising_train.txt
valid_file=$DATA_DIR/Advising_valid.txt
test_file=$DATA_DIR/Advising_test.txt
vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
embedded_vector_file=$DATA_DIR/glove_840B_300d_vec_plus_word2vec_100.txt

max_utter_len=30
max_utter_num=25
max_word_length=18
num_layer=1
embedding_dim=400
rnn_size=200

batch_size=100
lambda=0
dropout_keep_prob=0.5
num_epochs=60
evaluate_every=1000

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python -u ${PKG_DIR}/model/train_UCM.py \
                --train_file $train_file \
                --valid_file $valid_file \
                --test_file $test_file \
                --vocab_file $vocab_file \
                --char_vocab_file $char_vocab_file \
                --embedded_vector_file $embedded_vector_file \
                --max_utter_len $max_utter_len \
                --max_utter_num $max_utter_num \
                --max_word_length $max_word_length \
                --num_layer $num_layer \
                --embedding_dim $embedding_dim \
                --rnn_size $rnn_size \
                --batch_size $batch_size \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --num_epochs $num_epochs \
                --seeds 255 \
		--evaluate_every $evaluate_every >  UCM_log.txt 2>&1 &

