import torch
import torch.nn as nn


def BatchNorm2d(size):
    return nn.BatchNorm2d(size)


def InstanceNorm2d(size,affine=True):
    return nn.InstanceNorm2d(size,affine=True)

