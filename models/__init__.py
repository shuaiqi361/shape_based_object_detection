from .SSD300 import SSD300, MultiBoxLoss300
from .SSD512 import SSD512, MultiBoxLoss512
from .RetinaNet import resnet50, resnet101, RetinaFocalLoss
from .RefineDet512 import RefineDet512, RefineDetLoss
from .FCOSDet import resnet50_fcos, resnet101_fcos, FCOSLoss
from .SSD_PANet import SSDPANet, MultiBoxPANetLoss
from .RefineBOF import RefineDetBof, RefineDetBofLoss
from .RefineBOF2 import RefineDetBof2, RefineDetBofLoss2
from .RefineBOFTraffic import RefineDetBofTraffic, RefineDetBofTrafficLoss


def model_entry(config):
    if config.model['arch'].upper() == 'SSD300':
        print('Loading SSD300 with VGG16 backbone ......')
        return SSD300(config['n_classes'], device=config.device), MultiBoxLoss300
    elif config.model['arch'].upper() == 'SSD512':
        print('Loading SSD512 with VGG16 backbone ......')
        return SSD512(config['n_classes'], device=config.device), MultiBoxLoss512
    elif config.model['arch'].upper() == 'RETINA50':
        print('Loading RetinaNet with ResNet-50 backbone ......')
        return resnet50(config['n_classes'], config=config), RetinaFocalLoss
    elif config.model['arch'].upper() == 'RETINA101':
        print('Loading RetinaNet with ResNet-101 backbone ......')
        return resnet101(config['n_classes'], config=config), RetinaFocalLoss
    elif config.model['arch'].upper() == 'REFINEDET':
        print('Loading RefineDet with VGG-16(512) backbone ......')
        return RefineDet512(config['n_classes'], config=config), RefineDetLoss
    elif config.model['arch'].upper() == 'FCOS50':
        print('Loading Fully Convolutional One-Stage detector(FCOS) with ResNet-50 backbone ......')
        return resnet50_fcos(config['n_classes'], config=config), FCOSLoss
    elif config.model['arch'].upper() == 'FCOS101':
        print('Loading Fully Convolutional One-Stage detector(FCOS) with ResNet-101 backbone ......')
        return resnet101_fcos(config['n_classes'], config=config), FCOSLoss
    elif config.model['arch'].upper() == 'PANET':
        print('Loading PANET_FPN with VGG-16 backbone ......')
        return SSDPANet(config=config), MultiBoxPANetLoss
    elif config.model['arch'].upper() == 'REFINEDETBOF':
        print('Loading RefineDet with VGG-16 backbone, Bof augmented ......')
        return RefineDetBof(config['n_classes'], config=config), RefineDetBofLoss
    elif config.model['arch'].upper() == 'REFINEDETBOFTRAFFIC':
        print('Loading RefineDet with VGG-16 backbone, Bof augmented DETRAC finetune model ......')
        return RefineDetBofTraffic(config['n_classes'], config=config), RefineDetBofTrafficLoss
    elif config.model['arch'].upper() == 'REFINEDETBOF2':
        print('Loading RefineDet with VGG-16 backbone, Bof augmented v2 ......')
        return RefineDetBof2(config['n_classes'], config=config), RefineDetBofLoss2
    else:
        print('Try other models.')
        raise NotImplementedError
