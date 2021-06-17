import os
os.environ["OMP_NUM_THREADS"]="3"

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.train import train
from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from gan.losses import discriminator_loss, generator_loss

from gan.losses import ls_discriminator_loss, ls_generator_loss

from gan.models import Discriminator, Generator


batch_size = 128
scale_size = 64  # We resize the images to 64x64 for training

celeba_root = '../celeba_data'

celeba_train = ImageFolder(root=celeba_root, transform=transforms.Compose([
      transforms.Resize(scale_size),
        transforms.ToTensor(),
        ]))

celeba_loader_train = DataLoader(celeba_train, batch_size=batch_size, drop_last=True)

imgs = celeba_loader_train.__iter__().next()[0].numpy().squeeze()

NOISE_DIM = 100
NUM_EPOCHS = 50
learning_rate = 0.0002

D = Discriminator().to(device)
G = Generator(noise_dim=NOISE_DIM).to(device)

D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas = (0.5, 0.999))
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas = (0.5, 0.999))

# original gan
train(D, G, D_optimizer, G_optimizer, discriminator_loss,\
        generator_loss, num_epochs=NUM_EPOCHS, show_every=250,\
        train_loader=celeba_loader_train, device=device)
