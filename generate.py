import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random

import matplotlib.pyplot as plt
from PIL import Image

from models.DCGAN.models import *
from models.WGAN.models import *
from models.SRGAN.model import ResGenerator
from models_testing.image_rescaler import increase_img_resolution

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



standart_model_path = os.path.join(os.getcwd(), "models_testing", "pretrain_models", "animeDCGAN50" )
standart_output_path = os.path.join( os.getcwd(), "output" )
res_model_path = os.path.join(os.getcwd(), "models_testing", "pretrain_models", "SRGAN.pt" )

def generate(model_path=standart_model_path, output_size = 128, output_path = standart_output_path, use_upscaler=True, seed=random.randint(1, 1000)):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    if use_upscaler:
        res_generator = ResGenerator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16)
        res_generator.load_state_dict(torch.load(res_model_path, map_location=torch.device('cpu')))
        res_generator = res_generator.to(torch.device('cpu'))
        res_generator.eval()

    noise_size = 128
    unnormlizer = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    torch.manual_seed(seed)
    noise = torch.randn(1, noise_size, 1, 1).to(torch.device('cpu'))
    gen_img = model(noise).cpu().detach()
    gen_img = unnormlizer(gen_img)

    if use_upscaler:
        res_mode = output_size // 32
        gen_img = increase_img_resolution(gen_img, res_generator, res_mod=res_mode, split_img = False)
        res = transforms.Resize((output_size, output_size))
        gen_img = res(gen_img)
    else:
        print(gen_img.shape)
        res = transforms.Resize((output_size, output_size))
        gen_img = res(gen_img)
        

    gen_numpy = gen_img[0].permute(1,2,0).numpy()
    plt.figure(figsize=(8,8))
    plt.axis(False)
    plt.imshow(gen_numpy)
    plt.savefig(os.path.join(output_path, "output.jpeg"), bbox_inches='tight')
    return

if __name__ == "__main__":
    generate()