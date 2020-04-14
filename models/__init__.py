from .SSD300 import SSD300, MultiBoxLoss300
from .SSD512 import SSD512, MultiBoxLoss512


def model_entry(config):
    if config.model['arch'].upper() == 'SSD300':
        print('Loading SSD300 with VGG16 backbone ......')
        return SSD300(config['n_classes'], device=config.device), MultiBoxLoss300
    elif config.model['arch'].upper() == 'SSD512':
        print('Loading SSD512 with VGG16 backbone ......')
        return SSD512(config['n_classes'], device=config.device), MultiBoxLoss512
    else:
        print('Try other models.')
        raise NotImplementedError
