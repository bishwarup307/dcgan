from tqdm import tqdm
import git
import sys

import torch
import torchvision
import torchvision.transforms as trsf
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Critic, init_weights
from utils import (
    CutTop,
    AverageMeter,
    gen_noise,
    ModelCheckpoint,
    load_inception_model,
    matrix_sqrt,
    fid,
    inception_score,
    FeatureMap,
    get_inception_features,
    get_covariance,
    save_tensor,
    load_tensor,
)

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
LOG_FREQ = 500
# SPECTRAL_NORM = True
CKPT_FREQ = 500
KEEP_LAST_N_CKPT = 10

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

# dataset = torchvision.datasets.MNIST(
#     root="~/torchvision_datasets/MNIST", train=True, download=True, transform=transforms
# )
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True
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
    im_ch=IMAGE_CHANNELS,
    hidden_dim=HIDDEN_DIM_DISC,
    use_batchnorm=USE_BATCHNORM,
    spectral_norm=SPECTRAL_NORM,
)
critic = critic.to(device)

critic.apply(init_weights)
gen.apply(init_weights)

# configure loss and optimizers
criterion = nn.BCEWithLogitsLoss()
opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(beta1, beta2))
opt_disc = torch.optim.Adam(critic.parameters(), lr=LR, betas=(beta1, beta2))

# configure tensorboard writer
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:6]
logdir = f"/home/bishwarup/GAN_experiments/dcgan/{sha}"
writer = SummaryWriter(log_dir=logdir)

# make a fixed noise to see the generator evolve over time on it
fixed_noise = gen_noise(32, NOISE_DIM, device=device)

# train loop
gen.train()
critic.train()

checkpointer = ModelCheckpoint(logdir, freq=CKPT_FREQ, keep_n=KEEP_LAST_N_CKPT)

# initialize inception model for evaluation
inception = load_inception_model(
    pretrained=False,
    weights="/home/bishwarup/Downloads/inception_v3_google-1a9a5a14.pth",
    device=device,
)
get_avgpool = FeatureMap()
inception.avgpool.register_forward_hook(
    get_avgpool
)  # register hook to get avgpool output for FID

for epoch in range(EPOCHS):
    lossD = AverageMeter("LossD")
    lossG = AverageMeter("LossG")

    mu_real, sigma_real = load_tensor("mu", logdir), load_tensor("sigma", logdir)

    calculate_real_features = mu_real is None or sigma_real is None

    real_features = []
    fake_features = []  # required for FID
    fake_softmax = []  # required for inception score
    global_step = 0

    pbar = tqdm(enumerate(loader))
    for n_iter, (real, _) in pbar:

        real = real.to(device)
        # calculate global step
        global_step = len(loader) * epoch + n_iter

        # pass real images to inception for fid
        if calculate_real_features:
            _ = get_inception_features(inception, real)
            real_features.append(get_avgpool.output.detach().cpu())

        # calculate discriminator loss
        noise = gen_noise(BATCH_SIZE, NOISE_DIM, device=device)
        fake = gen(noise)

        logits = get_inception_features(inception, fake.detach())
        logits = F.softmax(logits, dim=1)
        fake_softmax.append(logits.detach().cpu())
        fake_features.append(get_avgpool.output.detach().cpu())

        disc_fake = critic(fake.detach())
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real = critic(real)
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_loss = (disc_loss_fake + disc_loss_real) / 2.0

        # update discriminator
        opt_disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        # monitor running loss
        lossD.update(disc_loss.item(), BATCH_SIZE)

        # calculate generator loss
        noise2 = gen_noise(BATCH_SIZE, NOISE_DIM, device=device)
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

        # write tensorboard every 500 step
        if global_step % LOG_FREQ == 0:
            gen.eval()
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise)
            grid = torchvision.utils.make_grid(fixed_fakes, normalize=True)
            # print(grid.size())
            writer.add_image("Generated fakes", grid, global_step=global_step)
            # sys.exit(0)
            gen.train()

        checkpointer.save(gen, global_step)

    if len(real_features):
        real_features = torch.cat(real_features, dim=0)
        mu_real = real_features.mean(dim=0)
        sigma_real = get_covariance(real_features)
        save_tensor(mu_real, "mu", logdir)
        save_tensor(sigma_real, "sigma", logdir)

    fake_features = torch.cat(fake_features, dim=0)
    fake_softmax = torch.cat(fake_softmax, dim=0)

    # calculate inception score
    inc_score = inception_score(fake_softmax)

    # calculate fid
    mu_fake = fake_features.mean(dim=0)
    sigma_fake = get_covariance(fake_features)
    fid_score = fid(mu_real, mu_fake, sigma_real, sigma_fake)

    writer.add_scalar("scores/FID", fid_score, global_step=global_step)
    writer.add_scalar("scores/IS", inc_score, global_step=global_step)

# if __name__ == "__main__":
#
#     print(sha)
