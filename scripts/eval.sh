cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data


test_file=$DATA_DIR/Advising_test.txt
vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
check_point=${parentdir}/scripts/runs/restore
max_utter_len=30
max_utter_num=25
max_word_length=18



PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -u ${PKG_DIR}/model/eval.py \
                --test_file $test_file \
                --vocab_file $vocab_file \
                --char_vocab_file $char_vocab_file \
                --max_utter_len $max_utter_len \
                --max_utter_num $max_utter_num \
                --max_word_length $max_word_length \
                --checkpoint_dir $check_point \
		            --output_file  output.txt

