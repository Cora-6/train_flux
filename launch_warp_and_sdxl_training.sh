export MODEL_DIR="black-forest-labs/FLUX.1-dev" #"stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="../data/output_flux_warp_large_dataset"
export WANDB_API_KEY=3eac3a8effee6e20659c4fe6c936861e0b44e7f2

accelerate launch --num_processes 2 train_controlnet_flux_warp.py\
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="../data/train" \
 --resolution=512 \
 --report_to="wandb" \
 --learning_rate=1e-5 \
 --num_train_epochs=48 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=32 \
 --gradient_checkpointing \
 --dataloader_num_workers 8 \
 --mixed_precision="bf16"
#  --resume_from_checkpoint='/data/output_sdxl_large_dataset_warp/checkpoint-32500'
#  --validation_image "/data/result_sphere/stitch/2025-03-07_11-20-44/render_h_transformed_hsi.mat" "/home/viplab/diffusers/examples/controlnet/mytry/dataset/result/2025-02-10_11-01-49"
#  --validation_prompt "city view"\
#  --max_train_steps=6000 \
