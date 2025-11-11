accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --gpu_ids "0,1" \
    train_rosa.py \
        --model_path "/home/hwj/Qwen3-0.6B" \
        --dataset_path "/home/hwj/mobvoi_seq_monkey_general_open_corpus__qwen3__ctx512" \
        --output_dir "output_sufa" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --adapter_type "sufa"
