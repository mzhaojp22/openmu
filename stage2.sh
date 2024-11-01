# pip install -e .

OUTPUT_DIR="./test_stage2"
mkdir -p $OUTPUT_DIR

export LLAMA3_PATH="pointer_to_your_llama3"
export AudioMAE_path="vitb_finetuned.pth"
export Stage2_training_data="MusicInstruct_long_eval.json" # please download OpenMU-Bench
export point_to_stage1_projector="stage1_projector.bin" # please download the checkpoint

declare -a list_of_loras=(16 128)
for ((pt=0;pt<${#list_of_loras[@]};pt+=2)); do
    python3 llava/train/train.py \
    --audio_num_pooling_tokens 8 \
    --lora_enable True --lora_r ${list_of_loras[pt]} --lora_alpha ${list_of_loras[pt+1]} \
    --mm_projector_lr 2e-5 \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $LLAMA3_PATH \
    --version v1 \
    --data_path $Stage2_training_data \
    --is_audio_exp \
    --image_folder images/ \
    --audio_tower mae_vit_base_patch16_dec512d8b \
    --vision_tower mae_vit_base_patch16_dec512d8b \
    --pretrain_mm_mlp_adapter $point_to_stage1_projector \
    --audio_pretrained_ckpt_path $AudioMAE_path \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --deepspeed ./scripts/zero2.json
done