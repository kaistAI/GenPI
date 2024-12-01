#!/bin/bash

# --adapter_checkpoint_dir ../checkpoints/stage1/context_r16_lr1e4_b16/checkpoint-93 \

NUM_EPOCH=10

STAGE2_RATIO=0.7
STAGE2_RATIO_FOR_PATH=${STAGE2_RATIO##*.}

STAGE3_RATIO=0.1
STAGE3_RATIO_FOR_PATH=${STAGE3_RATIO##*.}

CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file ../accelerate_config/1gpu.yaml --main_process_port 12345 ../../src/finetune.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ../../checkpoints/webshop/triple_loss/context_r16_lr1e4_b32_a32_gn5_stage2ratio${STAGE2_RATIO_FOR_PATH}_stage2ratio${STAGE3_RATIO_FOR_PATH}_stage123_triple_loss/ \
    --context_id 0 \
    --dataset agentbench-webshop \
    --dataset_format stage123_triple_loss \
    --source_max_len 512 \
    --target_max_len 512 \
    --logging_steps 10 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --max_steps -1 \
    --num_train_epochs ${NUM_EPOCH} \
    --data_seed 42 \
    --save_steps 27 \
    --save_total_limit ${NUM_EPOCH} \
    --evaluation_strategy no \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --max_new_tokens 512 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --max_grad_norm 0.5 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --lora_dropout 0.05 \
    --seed 0 \
    --max_memory_MB 80000 \
    --trust_remote_code True \
    --token True \
    --ddp_find_unused_parameters False \
    --stage2_ratio ${STAGE2_RATIO} \
    --stage3_ratio ${STAGE3_RATIO}
# done