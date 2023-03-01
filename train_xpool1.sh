batch_size=128
num_frames=16
num_prompts=8
num_test_frames=64
test_batch_size=32
num_epochs=20
noclip_lr=3e-5
clip_lr=1e-6
video_dir=data/ANet/ANet_Videos_12fps/

exp_name=anet_xpool_bs${batch_size}_nf${num_frames}_ep${num_epochs}_ema999_lr${noclip_lr}

python train.py \
    --exp_name=$exp_name \
    --videos_dir=$video_dir \
    --batch_size=$batch_size \
    --test_batch_size=$test_batch_size \
    --dataset_name=ActivityNet \
    --pooling_type=transformer \
    --pooling_type_test=transformer \
    --arch=prompt_clip \
    --num_frames=$num_frames \
    --num_test_frames=$num_test_frames \
    --num_prompts=$num_prompts \
    --num_workers=16 \
    --num_samples=8 \
    --loss='clip+caption' \
    --frequent_word_weight=0.2 \
    --num_captioner_layers=3 \
    --evals_per_epoch=1 \
    --num_epochs=$num_epochs \
    --noclip_lr=$noclip_lr \
    --clip_lr=$clip_lr \
    --use_ema \
    --model_ema_decay=0.999
