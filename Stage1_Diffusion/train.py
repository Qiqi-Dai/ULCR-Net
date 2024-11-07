import os
import numpy as np
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

Image_size = 128
Batch_size = 16
Num_sampling = 100
Data_folder = 'dataset/trainB'
Result_folder = 'results_dataset/noisydata'
Visual_folder = Result_folder + '/visual'

if not os.path.exists(Result_folder):
    os.makedirs(Result_folder)
if not os.path.exists(Visual_folder):
    os.makedirs(Visual_folder)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = Image_size,
    timesteps = 1000,   
    loss_type = 'l1'   # l1 or l2
).cuda()

#### Traning
trainer = Trainer(
    diffusion,
    Data_folder,
    train_batch_size = Batch_size,
    train_lr = 2e-5,
    train_num_steps = 20000,         # total training steps
    save_and_sample_every = 1000,
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    results_folder = Result_folder
)

trainer.train()

#### Sampling
for num in range(Num_sampling):
    sampled_images = diffusion.sample(batch_size = 1)
    sampled_images = np.array(sampled_images.cpu())
    print(sampled_images.shape) 
    for i in range(len(sampled_images)):
        pred = sampled_images[i].reshape(Image_size, Image_size) * 255
        pred = Image.fromarray(pred)
        pred.convert('L').save(Visual_folder + '/pred_%d_%d.png'%(num, i))
