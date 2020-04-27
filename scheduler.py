import json
import os
import torch


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale

    print("DECAYING learning rate, the new LR is %f" % (optimizer.param_groups[1]['lr'],))


def warm_up_learning_rate(optimizer, rate=4.5):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * rate

    print("WARMING up learning rate, the new LR is %f" % (optimizer.param_groups[1]['lr'],))
