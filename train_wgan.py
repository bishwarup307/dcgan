from tqdm import tqdm
import git
import numpy as np

import torch
import torchvision
import torchvision.transforms as trsf
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Critic, init_weights
from utils import CutTop, AverageMeter, gen_noise, get_grad, gradient_penalty

# Some settings
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
HIDDEN_DIM_GEN = 64
HIDDEN_DIM_DISC = 64
NOISE_DIM = 100
USE_BATCHNORM = True
UPSAMPLE_MODE = None
EPOCHS = 100
cur_batch_size = 128
WORKERS = 8
LR = 2e-4
beta1, beta2 = 0.5, 0.999
LOG_FREQ = 100
C_LAMBDA = 10
CRITIC_ITERS = 5

# initialize dataset and dataloader
transforms = trsf.Compose(
    [
        # CutTop(),
        # trsf.ToPILImage(),
        trsf.Resize(IMAGE_SIZE),
        trsf.ToTensor(),
        trsf.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
        ),
    ]
)
# dataset = torchvision.datasets.ImageFolder(
#     "/home/bishwarup/torchvision_datasets/celebA/img_align_celeba", transform=transforms
# )
dataset = torchvision.datasets.MNIST(
    root="~/torchvision_datasets/MNIST", train=True, download=True, transform=transforms
)

loader = DataLoader(
    dataset, batch_size=cur_batch_size, num_workers=WORKERS, pin_memory=True
)

# initialize models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen = Generator(
    im_ch=IMAGE_CHANNELS,
    latent_dim=NOISE_DIM,
    hidden_dim=HIDDEN_DIM_GEN,
    use_batchnorm=USE_BATCHNORM,
    upsample_mode=UPSAMPLE_MODE,
)
gen = gen.to(device)
critic = Critic(
    im_ch=IMAGE_CHANNELS, hidden_dim=HIDDEN_DIM_DISC, use_batchnorm=USE_BATCHNORM
)
critic = critic.to(device)

gen = gen.apply(init_weights)
critic = critic.apply(init_weights)

# configure loss and optimizers
criterion = nn.BCEWithLogitsLoss()
opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(beta1, beta2))
opt_disc = torch.optim.Adam(critic.parameters(), lr=LR, betas=(beta1, beta2))

# configure tensorboard writer
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:6]
writer = SummaryWriter(log_dir=f"/home/bishwarup/GAN_experiments/dcgan/{sha}")

# make a fixed noise to see the generator evolve over time on it
fixed_noise = gen_noise(32, NOISE_DIM, device=device)

# train loop
gen.train()
critic.train()

for epoch in range(EPOCHS):
    lossD = AverageMeter("LossD")
    lossG = AverageMeter("LossG")

    pbar = tqdm(enumerate(loader))
    for n_iter, (real, _) in pbar:

        real = real.to(device)
        cur_batch_size = real.size(0)

        # calculate global step
        global_step = len(loader) * epoch + n_iter

        crit_losses = []
        for _ in range(CRITIC_ITERS):
            # calculate discriminator loss
            noise = gen_noise(cur_batch_size, NOISE_DIM, device=device)
            fake = gen(noise)
            disc_fake = critic(fake.detach())
            disc_real = critic(real)

            # calculate gradient penalty
            epsilon = torch.rand(
                cur_batch_size, 1, 1, 1, device=device, requires_grad=True
            )
            grads = get_grad(critic, real, fake.detach(), epsilon)
            gp = gradient_penalty(grads)

            # calculate discriminator loss
            disc_loss = -(torch.mean(disc_real) - torch.mean(disc_fake)) + C_LAMBDA * gp

            # update discriminator
            opt_disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()
            crit_losses.append(disc_loss.item())

        # monitor running loss
        lossD.update(np.mean(crit_losses), cur_batch_size)

        # calculate generator loss
        noise2 = gen_noise(cur_batch_size, NOISE_DIM, device=device)
        fake2 = gen(noise2)
        disc_fake2 = critic(fake2)
        gen_loss = -torch.mean(disc_fake2)

        # update generator
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # monitor running loss
        lossG.update(gen_loss.item(), cur_batch_size)

        # update pbar description
        pbar.set_description(
            f"epoch: {epoch}, step: {global_step}, lossG: {gen_loss:.4f}, lossD: {disc_loss:.4f}"
        )

        # write tensorboard every 500 step
        if global_step % LOG_FREQ == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise)
            grid = torchvision.utils.make_grid(fixed_fakes, normalize=True)
            # print(grid.size())
            writer.add_image("Generated fakes", grid, global_step=global_step)
            # sys.exit(0)

# if __name__ == "__main__":
#
#     print(sha)
