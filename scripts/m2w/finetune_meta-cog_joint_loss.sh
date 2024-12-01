#!/bin/bash

# --adapter_checkpoint_dir ../checkpoints/stage1/context_r16_lr1e4_b16/checkpoint-93 \

NUM_EPOCH=10
STAGE2_RATIO=0.3

STAGE2_RATIO_FOR_PATH=${STAGE2_RATIO##*.}

CUDA_VISIBLE_DEVICES="0" WANDB_PROJECT=piv2 accelerate launch --config_file ../accelerate_config/1gpu.yaml --main_process_port 12581 ../../src/finetune.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ../../checkpoints/m2w/joint_loss_2/context_r16_lr1e4_b32_a32_gn5_ratio${STAGE2_RATIO_FOR_PATH}_cot-stage1_conv_stage2_joint_loss/ \
    --context_id 0 \
    --dataset agentbench-m2w \
    --dataset_format cot-stage1_conv_stage2_joint_loss \
    --source_max_len 512 \
    --target_max_len 512 \
    --logging_steps 10 \
    --save_strategy epoch \
    --logging_strategy steps \
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
    --report_to wandb \
    --run_name m2w_joint_${STAGE2_RATIO} \
    --stage2_ratio ${STAGE2_RATIO}
# done