import pprint
import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.utils.data as data
import time
import yaml
from tensorboardX import SummaryWriter
import argparse
from easydict import EasyDict
import json

from scheduler import adjust_learning_rate, WarmUpScheduler
from models import model_entry
from dataset.Datasets import PascalVOCDataset, COCO17Dataset, BaseModelVOCOCODataset
from utils import create_logger, save_checkpoint
from models.utils import detect
from metrics import AverageMeter, calculate_mAP
from dataset.transforms import bof_augment

parser = argparse.ArgumentParser(description='PyTorch 2D object detection training script.')
parser.add_argument('--config', default='', type=str)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--save-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--finetune', action='store_true')


def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    config = EasyDict(config)

    config.save_path = os.path.join(os.path.dirname(args.config), 'snapshots')
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    config.log_path = os.path.join(os.path.dirname(args.config), 'logs')
    if not os.path.exists(config.log_path):
        os.mkdir(config.log_path)
    config.event_path = os.path.join(os.path.dirname(args.config), 'events')
    if not os.path.exists(config.event_path):
        os.mkdir(config.event_path)

    with open(config.label_path, 'r') as j:
        config.label_map = json.load(j)

    config.n_classes = len(config.label_map)  # number of different types of objects

    iterations = config.optimizer['max_iter']
    workers = config.workers
    val_freq = config.val_freq
    lr = config.optimizer['base_lr']
    # decay_lr_at = [it * config.num_iter_flag for it in config.optimizer['decay_iter']]
    #
    # decay_lr_to = config.optimizer['decay_lr']
    momentum = config.optimizer['momentum']
    weight_decay = config.optimizer['weight_decay']
    if torch.cuda.device_count() < 2:
        config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if config.device == 1:  # only for 2 GPUs max
            config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data_folder = config.train_data_root
    val_data_folder = config.val_data_root
    if not isinstance(config.model['input_size'], list):
        input_size = (int(config.model['input_size']), int(config.model['input_size']))

    # Learning parameters
    if args.recover:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        start_epoch = checkpoint['epoch'] + 1
        print('Resume training from checkpoint %s epoch %d.\n' % (args.load_path, start_epoch))
        model = checkpoint['model']
        _, criterion = model_entry(config)
        optimizer = checkpoint['optimizer']
    else:
        start_epoch = 0
        model, criterion = model_entry(config)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        if config.optimizer['type'].upper() == 'SGD':
            optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                        lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif config.optimizer['type'].upper() == 'ADAM':
            optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                         lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError

    # Custom dataloaders
    if config.data_name.upper() == 'COCO':
        train_dataset = COCO17Dataset(train_data_folder, split='train', config=config)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, num_workers=workers,
                                                   pin_memory=False, drop_last=True)
        test_dataset = COCO17Dataset(val_data_folder, split='val', config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    elif config.data_name.upper() == 'VOC':
        train_dataset = PascalVOCDataset(train_data_folder, split='train', input_size=input_size, config=config)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, num_workers=workers,
                                                   pin_memory=False, drop_last=True)
        test_dataset = PascalVOCDataset(val_data_folder, split='val', input_size=input_size, config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    elif config.data_name.upper() == 'VOCOCO':
        train_dataset = BaseModelVOCOCODataset(train_data_folder, split='train', config=config)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, num_workers=workers,
                                                   pin_memory=False, drop_last=True)
        test_dataset = BaseModelVOCOCODataset(val_data_folder, split='val', config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)

    else:
        raise NotImplementedError

    if args.evaluate:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        print('Evaluate model from checkpoint %s at epoch %d.\n' % (args.load_path, start_epoch))
        model = checkpoint['model']
        saved_epoch = checkpoint['epoch']
        model = model.to(config.device)
        optimizer = checkpoint['optimizer']

        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_H-%M-%S")
        config.logger = create_logger('global_logger', os.path.join(config.log_path,
                                                                    'eval_result_{}_{}.txt'.format(config.model['arch'],
                                                                                                   date_time)))
        print('Length of Testing Dataset:', len(test_dataset))
        print('evaluate checkpoint: ', args.load_path, ' at epoch: ', saved_epoch)
        evaluate(test_loader, model, optimizer, config=config)

    cudnn.benchmark = True
    model = model.to(config.device)
    criterion = criterion(priors_cxcy=model.priors_cxcy, config=config).to(config.device)

    # create logger to track training results
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_H-%M-%S")
    config.tb_logger = SummaryWriter(config.event_path)
    config.logger = create_logger('global_logger', os.path.join(config.log_path,
                                                                'log_{}_{}.txt'.format(config.model['arch'],
                                                                                       date_time)))
    config.logger.info('args: {}'.format(pprint.pformat(args)))
    config.logger.info('config: {}'.format(pprint.pformat(config)))

    epochs = iterations // (len(train_dataset) // config.batch_size)

    print('total train epochs: ', epochs, ' training starts ......')
    str_print = 'Dataset size: {}'.format(len(train_dataset))
    config.logger.info(str_print)

    # Epochs
    best_mAP = -1.
    config.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, config.optimizer['min_lr'])

    for epoch in range(start_epoch, epochs):
        config.tb_logger.add_scalar('learning_rate', epoch)

        # evaluate(test_loader, model, optimizer, config=config)

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch, config=config)

        config.scheduler.step()

        # Save checkpoint
        if (epoch > 0 and epoch % val_freq == 0) or epoch == 2:
            _, current_mAP = evaluate(test_loader, model, optimizer, config=config)
            config.tb_logger.add_scalar('mAP', current_mAP, epoch)
            if current_mAP > best_mAP:
                save_checkpoint(epoch, model, optimizer,
                                name='{}/{}_{}_checkpoint_epoch-{}.pth.tar'.format(config.save_path,
                                                                                   config.model['arch'].lower(),
                                                                                   config.data_name.lower(), epoch))
                best_mAP = current_mAP

    # Save the last checkpoint if it is better
    _, current_mAP = evaluate(test_loader, model, optimizer, config=config)
    config.tb_logger.add_scalar('mAP', current_mAP, epoch)
    if current_mAP > best_mAP:
        save_checkpoint(epoch, model, optimizer,
                        name='{}/{}_{}_checkpoint_epoch-{}.pth.tar'.format(config.save_path,
                                                                           config.model['arch'].lower(),
                                                                           config.data_name.lower(), epoch))


def train(train_loader, model, criterion, optimizer, epoch, config):
    """
    One epoch's training.

    :param config:
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    # torch.cuda.empty_cache()
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    optimizer.zero_grad()
    if epoch == 0 and config.optimizer['warm_up']:
        lr_warmup = WarmUpScheduler(config.optimizer['base_lr'], config.optimizer['warm_up_steps'], optimizer)

    for i, (images, boxes, labels, _, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        if config.optimizer['warm_up'] and epoch == 0 and i % config.optimizer['warm_up_freq'] == 0 and i > 0:
            lr_warmup.update()

        data_time.update(time.time() - start)
        optimizer.zero_grad()
        # Bag of Freebies
        images, boxes, labels = bof_augment(images, boxes, labels, config=config)

        # Move to default device
        images = torch.stack(images, dim=0).to(config.device)
        boxes = [b.to(config.device) for b in boxes]
        labels = [l.to(config.device) for l in labels]

        # Forward prop.
        odm_locs, odm_scores = model(images)

        # Loss
        loss = criterion(odm_locs, odm_scores, boxes, labels)

        # Backward prop.
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % config.print_freq == 0:
            str_print = 'Epoch: [{0:3d}][{1}/{2}]\t' \
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses)
            config.logger.info(str_print)

    config.tb_logger.add_scalar('training_loss', losses.avg, epoch)

    del odm_locs, odm_scores, images, boxes, labels


def evaluate(test_loader, model, optimizer, config):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    # torch.cuda.empty_cache()
    model.eval()

    pp = pprint.PrettyPrinter()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    detect_speed = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, _, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = torch.stack(images, dim=0).to(config.device)
            boxes = [b.to(config.device) for b in boxes]
            labels = [l.to(config.device) for l in labels]
            difficulties = [d.to(config.device) for d in difficulties]

            # Forward prop.
            time_start = time.time()
            predicted_locs, predicted_scores = model(images)

            if config.data_name.upper() == 'COCO' or config.data_name.upper() == 'VOCOCO':
                det_boxes_batch, det_labels_batch, det_scores_batch = \
                    detect(predicted_locs,
                           predicted_scores,
                           min_score=config.nms['min_score'],
                           max_overlap=config.nms['max_overlap'],
                           top_k=config.nms['top_k'], priors_cxcy=model.priors_cxcy,
                           config=config)
            elif config.data_name.upper() == 'VOC':
                det_boxes_batch, det_labels_batch, det_scores_batch = \
                    detect(predicted_locs,
                           predicted_scores,
                           min_score=config.nms['min_score'],
                           max_overlap=config.nms['max_overlap'],
                           top_k=config.nms['top_k'], priors_cxcy=model.priors_cxcy,
                           config=config)
            else:
                raise NotImplementedError

            time_end = time.time()
            # for fair comparision with the paper's results and other repos

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            detect_speed.append((time_end - time_start) / len(labels))

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.5,
                                 config.label_map, config.device)

    # Print AP for each class
    pp.pprint(APs)

    # # added to resume training
    # model.train()

    str_print = 'EVAL: Mean Average Precision {0:.3f}, ' \
                'avg speed {1:.2f} Hz, lr {2:.6f}'.format(mAP, 1. / np.mean(detect_speed),
                                                          config.scheduler.get_lr()[1])
    config.logger.info(str_print)

    del predicted_locs, predicted_scores, boxes, labels, images, difficulties

    return APs, mAP


if __name__ == '__main__':
    main()
