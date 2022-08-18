import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import matplotlib.pyplot as plt

from models.DCGAN.models import DC_Discriminator, DC_Generator, initialize_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
small_batch_size = 64

img_dim = 64
num_channels = 3
num_epochs = 50

noise_dim_dcgan = 128
features_disc = 64
features_gen_dcgan = 64
lr_dc = 2e-4



generator_dc = DC_Generator(noise_dim_dcgan, num_channels, features_gen_dcgan).to(device)
discriminator_dc = DC_Discriminator(num_channels, features_disc).to(device)
initialize_weights(generator_dc)
initialize_weights(discriminator_dc)

opt_gen_dc = optim.Adam(generator_dc.parameters(), lr=lr_dc, betas=(0.5, 0.999))
opt_disc_dc = optim.Adam(discriminator_dc.parameters(), lr=lr_dc, betas=(0.5, 0.999))
criterion = nn.BCELoss()

writer_real = SummaryWriter("./AnimeFaceDC/real")
writer_fake = SummaryWriter("./AnimeFaceDC/fake")

generator_dc.train()
discriminator_dc.train()



def train_gan(disc, gen, loader, batch_size=128, epoch_verbose = 1, num_epochs = 10):
    step = 0
    fixed_noise = torch.randn(batch_size, noise_dim_dcgan, 1, 1).to(device)
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn(batch_size, noise_dim_dcgan, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc_dc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen_dc.step()

        # Print losses occasionally and print to tensorboard
        if epoch % epoch_verbose == 0:
            gen.eval()
            disc.eval()
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
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

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            disc.train()
    torch.save(gen, "model_dc")
    return gen