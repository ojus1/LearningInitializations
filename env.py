import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
import math

from target_model import generate_model, train_target, evaluate


class Normalizer:
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.state = {'mean': 0.0, 'var': 0.0, 'cum': 0.0, 'step': 0}

    def __call__(self, x):
        self.state['step'] += 1
        self.state['cum'] += x

        self.state['mean'] = self.alpha * \
            self.state['mean'] + (1 - self.alpha) * x
        self.state['var'] = self.alpha * \
            self.state['var'] + (1 - self.alpha) * x * x

    def normalize(self, x):
        return (x - self.state['mean'] / math.sqrt(self.state['var'] + 1e-8))

    def ema(self):
        return self.state['mean']

    def mean(self):
        return self.state['cum'] / self.state['step']


class Env:
    def __init__(self, dataset_name, training_steps=100):
        assert(dataset_name in ['mnist', 'fashion', 'cifar10'])
        self.training_steps = training_steps
        if dataset_name == 'mnist':
            dm_cls = MNIST
        elif dataset_name == 'fashion':
            dm_cls = FashionMNIST
        elif dataset_name == 'cifar10':
            dm_cls = CIFAR10
        self.dataset_name = dataset_name

        dataset = dm_cls("./data/", download=True,
                         transform=transforms.ToTensor(), train=True)
        val_size = int(0.84 * len(dataset))
        train_size = len(dataset) - val_size

        self.train, self.val = random_split(dataset, [train_size, val_size])
        self.test = dm_cls("./data/", download=True,
                           transform=transforms.ToTensor(), train=False)
        self.reward_normalizer = Normalizer(0.3)

    def dataloaders(self):
        train_dl = DataLoader(self.train, batch_size=256,
                              shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(self.val, batch_size=256,
                            shuffle=True, num_workers=2, pin_memory=True)
        test_dl = DataLoader(self.test, batch_size=256,
                             shuffle=True, num_workers=2, pin_memory=True)
        return train_dl, val_dl, test_dl

    def reset(self):
        if self.dataset_name == 'cifar10':
            return generate_model(3 * 32 * 32, 10, random.randint(1, 10))
        return generate_model(784, 10, random.randint(1, 10))

    def step(self, model):
        metrics = train_target(model, *self.dataloaders(), self.training_steps)
        reward = sum(metrics) / 3
        self.reward_normalizer(reward)
        return self.reward_normalizer.normalize(reward)


if __name__ == "__main__":
    e = Env(10)
    model = e.reset()
    print(e.step(model))
