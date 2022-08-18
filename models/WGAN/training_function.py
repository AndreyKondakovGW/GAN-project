import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import matplotlib.pyplot as plt

from models.WGAN.models import WGAN_Discriminator, WGAN_Generator, initialize_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
small_batch_size = 64

img_dim = 64
num_channels = 3
num_epochs = 50

noise_dim_wgan = 128
features_critic = 64
features_gen_wgan = 64
critic_iter = 5
weight_clip = 0.01
lr_w = 5e-5

generator_w = WGAN_Generator(noise_dim_wgan, num_channels, features_gen_wgan).to(device)
critic_w = WGAN_Discriminator(num_channels, features_critic).to(device)
initialize_weights(generator_w)
initialize_weights(critic_w)

opt_gen_w = optim.RMSprop(generator_w.parameters(), lr=lr_w)
opt_critic_w = optim.RMSprop(critic_w.parameters(), lr=lr_w)

writer_real_w = SummaryWriter(f"./AnimeFace_WGAN/real")
writer_fake_w = SummaryWriter(f"./AnimeFace_WGAN/fake")

generator_w.train()
critic_w.train()

def train_wgan(critic, gen, loader, batch_size=128, epoch_verbose = 1, num_epochs = 10):
    step = 0
    fixed_noise = torch.randn(batch_size, noise_dim_wgan, 1, 1).to(device)
    for epoch in tqdm(range(num_epochs)):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            for i in range(critic_iter):
                noise = torch.randn(batch_size, noise_dim_wgan, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic_w.step()
                # clip critic weights between -0.01, 0.01
                for p in critic.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)
                
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen_w.step()
        # Print losses occasionally and print to tensorboard
        if epoch % epoch_verbose == 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                numpy_grid = img_grid_fake.permute(1, 2, 0).cpu().detach().numpy()
                plt.axis(False)
                plt.imshow(numpy_grid)
                plt.show()

                writer_real_w.add_image("Real", img_grid_real, global_step=step)
                writer_fake_w.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
    torch.save(gen, "model_wgan")
    return gen