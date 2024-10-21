OUTPUT_DIR=""
mkdir -p $OUTPUT_DIR

# sif container
EXAMPLE_CONTAINER_NAME="openmu.sif"
CONTAINER_SRC_PATH="$EXAMPLE_CONTAINER_NAME"
LLAMA3_PATH=""
declare -a list_of_pooltokens=(8)

for ((pt=0;pt<${#list_of_pooltokens[@]};++pt)); do
   time mpirun -x NCCL_SOCKET_IFNAME=ib \
      python3 \
      llava/train/train_mem.py \
      --audio_num_pooling_tokens ${list_of_pooltokens[pt]} \
      --output_dir $OUTPUT_DIR \
      --model_name_or_path $LLAMA3_PATH \
      --version plain \
      --is_audio_exp \
      --audio_tower mae_vit_base_patch16_dec512d8b \
      --vision_tower mae_vit_base_patch16_dec512d8b \
      --data_path SoniDoMU_captioning.json \
      --image_folder images/ \
      --audio_pretrained_ckpt_path vitb_finetuned.pth \
      --mm_projector_type mlp2x_gelu \
      --tune_mm_mlp_adapter True \
      --mm_vision_select_layer -1 \
      --mm_use_im_start_end False \
      --mm_use_im_patch_token False \
      --bf16 True \
      --num_train_epochs 15 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 1000 \
      --save_total_limit 1 \
      --learning_rate 1e-3 \
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
      --audio_input_target_length 3072 \
      --deepspeed point_to_zero3.json
done      