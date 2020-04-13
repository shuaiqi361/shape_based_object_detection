from .SSD300 import SSD300, MultiBoxLoss300


def model_entry(config):
    if config.model['arch'].upper() == 'SSD300':
        print('Loading SSD300 with VGG16 backbone ......')
        return SSD300(config['n_classes'], device=config.device), MultiBoxLoss300
    else:
        print('Try other models.')
        raise NotImplementedError
