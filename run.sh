#!/bin/bash

i=0
model_path="/root/autodl-tmp/bert-models/Salesforce/SFR-Embedding-Mistral"
model_version="zero_round"
lora_path="none"
# CUDA_VISIBLE_DEVICES=$i nohup python3 -u recall.py ${model_path} ${model_version} ${lora_path} > log_recall_zero 2>&1


PATH_PRE="output/"
DATA_NAME="zero_round_recall_top_100_train.jsonl"
DATA_DIR=${PATH_PRE}
MODEL_USE="v3_round1"
ZERO_STAGE=2
OUTPUT=./model_save/${MODEL_USE}_qlora_rerun


mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
# nohup deepspeed  --master_port ${MASTER_PORT}  --include localhost:0 simcse_deepspeed_mistrial_qlora_argu.py \
#        --project_name ${name}_${MODEL_USE} \
#        --train_data ${DATA_DIR}${DATA_NAME} \
#        --model_name_or_path ${model_path} \
#        --per_device_train_batch_size 2 \
#        --per_device_eval_batch_size 2 \
#        --train_group_size 4 \
#        --gradient_accumulation_steps 64 \
#        --query_max_len 256 \
#        --passage_max_len 256 \
#        --earystop 0 \
#        --save_batch_steps 100000000000 \
#        --eary_stop_epoch 5 \
#        --save_per_epoch 1 \
#        --num_train_epochs 20 \
#        --learning_rate 1e-4 \
#        --num_warmup_steps 100 \
#        --weight_decay 0.01 \
#        --lr_scheduler_type cosine \
#        --seed 1234 \
#        --zero_stage $ZERO_STAGE \
#        --deepspeed \
#        --output_dir $OUTPUT \
#        --gradient_checkpointing  > log_simcse 2>&1


model_version="v3_round1_qlora"
lora_path="./model_save/${model_version}_rerun/epoch_19_model/adapter.bin"
# CUDA_VISIBLE_DEVICES=$i nohup python3 -u recall.py ${model_path} ${model_version} ${lora_path} > log_recall_train 2>&1


cd FlagEmbedding-master/FlagEmbedding/llm_reranker/finetune_for_instruction/

model_name_or_path="/root/autodl-tmp/bert-models/upstage/SOLAR-10.7B-Instruct-v1.0"
DATA_NAME="v3_round1_qlora_rank_top_50_train"
DATA_DIR="../../../../output/"
MODEL_USE="v3_round1_qlora_recall_top_100_for_rank_model"
OUTPUT="../../../../model_save/${MODEL_USE}"

export CUDA_VISIBLE_DEVICES=0
nohup torchrun --nproc_per_node 1 \
-m run \
--output_dir ${OUTPUT} \
--model_name_or_path ${model_name_or_path} \
--train_data ${DATA_DIR}${DATA_NAME}.jsonl \
--learning_rate 2e-4 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 64 \
--dataloader_drop_last True \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 16 \
--logging_steps 1 \
--save_total_limit 0 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage2.json \
--report_to "none" \
--warmup_ratio 0.05 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head > log 2>&1


cd ../../../../
rank_lora_path="./model_save/$MODEL_USE/adapter.bin"
# CUDA_VISIBLE_DEVICES=$i nohup python3 -u get_test_rank_result.py ${model_name_or_path} ${rank_lora_path} > log_rank 2>&1
