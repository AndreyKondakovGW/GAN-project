from torch import nn

class Discriminator(nn.Module):
  '''принимает на вход плоские вектор размера in_features,
      возвращает число от 0 до 1 '''
  def __init__(self, in_features):
      super().__init__()
      self.disc = nn.Sequential(
          nn.Linear(in_features, 128),
          nn.LeakyReLU(0.01),
          nn.Linear(128, 1),
          nn.Sigmoid(),
      )

  def forward(self, x):
      return self.disc(x)

class Generator(nn.Module):
  '''принимает на вход плоские вектор размера z_dim(шум),
  возвращает изображение(вектор) размера img_dim'''
  def __init__(self, z_dim, img_dim):
    super().__init__()
    self.gen = nn.Sequential(
      nn.Linear(z_dim, 256),
      nn.LeakyReLU(0.01),
      nn.Linear(256, img_dim),
      nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
    )

  def forward(self, x):
      return self.gen(x)