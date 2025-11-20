#!/bin/bash



model_name_or_path="/data_local/bert-models/upstage/SOLAR-10.7B-Instruct-v1.0"
# model_name_or_path="/data_local/bert-models/unsloth/gemma-2-9b-it-bnb-4bit"
# model_name_or_path="/data_local/bert-models/Qwen/Qwen2.5-7B-Instruct"
DATA_NAME="v3_round1_qlora_rank_top_50_train"
DATA_DIR="../../../../output/"
MODEL_USE="v3_round1_qlora_recall_top_100_for_rank_model"
OUTPUT=../../../../model_save/${MODEL_USE}

export CUDA_VISIBLE_DEVICES=2,3
nohup torchrun --nproc_per_node 2 \
-m run \
--output_dir ${OUTPUT} \
--model_name_or_path ${model_name_or_path} \
--train_data ${DATA_DIR}${DATA_NAME}.jsonl \
--learning_rate 2e-4 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--dataloader_drop_last True \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 16 \
--logging_steps 1 \
--save_strategy epoch \
--save_steps 1 \
--save_total_limit 50 \
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
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head > log 2>&1 &