Implementation codes and datasets for the paper "Learning from Clutter: An Unsupervised Learning-Based Clutter Removal Scheme for GPR B-Scans"

1. For the diffusion model used for data augmentation in Stage 1, we refer to the usage at https://github.com/lucidrains/denoising-diffusion-pytorch. Get into the folder "Stage1_Diffusion" and run "python train.py".

2. For the CUT model adopted for clutter estimation in stage 2, we used the codes at https://github.com/taesungp/contrastive-unpaired-translation. Get into the folder "Stage2_CUT" and run the following commands:

    python train.py \
    --dataroot ./dataset \
    --name dataset \
    --model cut \
    --preprocess none \
    --n_epochs 200 \
    --n_epochs_decay 200 \
    --load_size 128 \
    --crop_size 128
