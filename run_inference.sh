# pip install -e .

export LLAMA3_path="pointer_to_your_llama3"
export AudioMAE_path="pointer_to_the_audio_mae_path"
export ckpt_path="OpenMU_checkpoint_path"


python3 llava/eval/model_lyrics_grid.py \
        --model-base $LLAMA3_path \
        --audio-ckpt $AudioMAE_path \
        --model-path $ckpt_path \
        --answers-file mypredict.jsonl \
        --num_beams 5 \
        --top_p 0.9 \
        --max_new_tokens 512