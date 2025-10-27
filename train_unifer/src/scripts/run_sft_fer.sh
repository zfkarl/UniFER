
cd ./UniFER/train_unifer/src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_sft.txt"

source /miniconda3/bin/activate

conda activate r1-v

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_fer.py \
    --output_dir "./UniFER/model/Qwen2.5-VL-7B-FER-COT-SFT-230K" \
    --model_name_or_path "./UniFER/model/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "./UniFER/data/UniFER_CoT_230K.json" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --bf16 True \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name Qwen2.5-VL-7B-FER-COT-SFT-230K \
    --save_steps 20000 \
    --max_grad_norm 5 \
    --save_only_model true \