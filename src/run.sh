# -*- coding: utf-8 -*-
ts=`date +%Y%m%d%H%M`
data_sign=conv
data_dir=../data
output_dir=../${ts}

config_path=./config/bert.json
bert_model=/home/qhj/.lensnlp/language_model/bert-base-chinese.tar.gz 
vocab=../raw/vocab.txt

task_name=clf
max_seq_len=64
train_batch=64
dev_batch=64
test_batch=64
learning_rate=2e-5
num_train_epochs=4
warmup=0.1
local_rank=-1
seed=3306
checkpoint=50
log_dir=../${ts}

python generate_dict.py
python preprocess.py

CUDA_VISIBLE_DEVICES=1 python run.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--vocab ${vocab} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--train_batch_size ${train_batch} \
--dev_batch_size ${dev_batch} \
--test_batch_size ${test_batch} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--output_dir ${output_dir} \
--log_path ${log_dir}
