from tqdm import tqdm
import git

import torch
import torchvision
import torchvision.transforms as trsf
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Critic
from utils import CutTop, AverageMeter, gen_noise

# Some settings
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
HIDDEN_DIM_GEN = 64
HIDDEN_DIM_DISC = 64
NOISE_DIM = 100
USE_BATCHNORM = True
UPSAMPLE_MODE = None
EPOCHS = 100
BATCH_SIZE = 128
WORKERS = 8
LR = 2e-4
beta1, beta2 = 0.5, 0.999

# initialize dataset and dataloader
transforms = trsf.Compose(
    [
        CutTop(),
        trsf.ToPILImage(),
        trsf.Resize(IMAGE_SIZE),
        trsf.ToTensor(),
        trsf.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
        ),
    ]
)
dataset = torchvision.datasets.ImageFolder(
    "/home/bishwarup/torchvision_datasets/celebA/img_align_celeba", transform=transforms
)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True
)

# initialize models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen = Generator(
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

# configure loss and optimizers
criterion = nn.BCEWithLogitsLoss()
opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(beta1, beta2))
opt_disc = torch.optim.Adam(critic.parameters(), lr=LR, betas=(beta1, beta2))

# configure tensorboard writer
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:6]
writer = SummaryWriter(log_dir=f"/home/bishwarup/GAN_experiments/dcgan/{sha}")

# make a fixed noise to see the generator evolve over time on it
fixed_noise = gen_noise(32, NOISE_DIM)

# train loop
gen.train()
critic.train()

for epoch in range(EPOCHS):
    lossD = AverageMeter("LossD")
    lossG = AverageMeter("LossG")

    pbar = tqdm(enumerate(loader), ncols=80)
    for n_iter, (real, _) in pbar:

        # calculate global step
        global_step = len(loader) * epoch + n_iter

        # calculate discriminator loss
        noise = gen_noise(BATCH_SIZE, NOISE_DIM)
        fake = gen(noise).to(device)
        disc_fake = critic(fake.detach())
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real = critic(real.to(device))
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_loss = (disc_loss_fake + disc_loss_real) / 2.0

        # update discriminator
        opt_disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        # monitor running loss
        lossD.update(disc_loss.item(), BATCH_SIZE)

        # calculate generator loss
        noise2 = gen_noise(BATCH_SIZE, NOISE_DIM)
        fake2 = gen(noise2)
        disc_fake2 = critic(fake2)
        gen_loss = criterion(disc_fake2, torch.ones_like(disc_fake2))

        # update generator
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # monitor running loss
        lossG.update(gen_loss.item(), BATCH_SIZE)

        # update pbar description
        pbar.set_description(
            f"epoch: {epoch}, step: {global_step}, lossG: {gen_loss:.4f}, lossD: {disc_loss:.4f}"
        )

        # write tensorboard
        with torch.no_grad():
            fixed_fakes = gen(fixed_noise)
            grid = torchvision.utils.make_grid(fixed_fakes)
            writer.add_image("Generated fakes", grid, global_step=global_step)

# if __name__ == "__main__":
#
#     print(sha)
