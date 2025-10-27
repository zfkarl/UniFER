#!/bin/bash



cd ./UniFER/train_unifer/src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_grpo.txt"

OUTPUT_DIR="./log/Qwen2.5-VL-7B-FER-GRPO-VLLM-8GPU"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi

DS_CONFIG="local_scripts/zero3.json"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

source /miniconda3/bin/activate

conda activate r1-v

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir './Qwen2.5-VL-7B-FER-GRPO-VLLM-8GPU' \
    --model_name_or_path './UniFER/model/Qwen2.5-VL-7B-FER-COT-SFT-230K' \
    --dataset_name "./UniFER/data/UniFER_RLVR_360K.json" \
    --max_prompt_length 4096 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 2 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "Qwen2.5-VL-7B-FER-GRPO-VLLM-8GPU" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_only_model True \
    --report_to wandb \
    --beta 0.04 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 8 \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.8 \
    --deepspeed local_scripts/zero3.json \
    2>&1 | tee "./log/Qwen2.5-VL-7B-FER-GRPO-VLLM-8GPU/training_log.txt"




