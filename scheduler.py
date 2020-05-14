import json
import os
import torch
import math


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale

    print("DECAYING learning rate, the new LR is %f" % (optimizer.param_groups[1]['lr'],))


def warm_up_learning_rate(optimizer, rate=5.0):
    """
    Scale learning rate by a specified factor.

    :param rate:
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * rate

    print("WARMING up learning rate, the new LR is %f" % (optimizer.param_groups[1]['lr'],))


class WarmUpScheduler(object):
    def __init__(self, target_lr, n_steps, optimizer):
        self.target_lr = target_lr
        self.init_lr = target_lr / (2. ** n_steps)
        self.n_steps = n_steps
        self.rate = 2.  # math.log(target_lr / init_lr, n_steps)
        self.optimizer = optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / (2 ** n_steps)

    def update(self):
        if self.n_steps > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.rate
        else:
            return

        self.n_steps -= 1

