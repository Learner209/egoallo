import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
import math

np.set_printoptions(precision=30,  floatmode='maxprec')
prec = 10


def equal(a0, a1):
    return np.array_equal(np.round(a0, prec), np.round(a1, prec))

def ewma(x, alpha = 0.05):
    avg = x[0]
    for i in x[1:]:
        avg = alpha * i + (1- alpha) * avg
    return avg

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

if __name__ == "__main__":
    pass