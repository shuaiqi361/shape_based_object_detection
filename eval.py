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

from scheduler import adjust_learning_rate
from models import model_entry
from dataset.Datasets import PascalVOCDataset, COCO17Dataset
from utils import create_logger, save_checkpoint, clip_gradient
from models.utils import detect
from metrics import AverageMeter, calculate_mAP

parser = argparse.ArgumentParser(description='PyTorch 2D object detection training script.')
parser.add_argument('--config', default='', type=str)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--model-type', default='anchor/refine/point/shape', type=str)
parser.add_argument('-e', '--evaluate', action='store_true')


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

    batch_size = config.batch_size
    config.num_iter_flag = batch_size // config.internal_batchsize
    with open(config.label_path, 'r') as j:
        config.label_map = json.load(j)

    config.n_classes = len(config.label_map)  # number of different types of objects

    iterations = config.optimizer['max_iter'] * config.num_iter_flag
    workers = config.workers
    val_freq = config.val_freq
    lr = config.optimizer['base_lr']
    decay_lr_at = [it * config.num_iter_flag for it in config.optimizer['decay_iter']]

    decay_lr_to = config.optimizer['decay_lr']
    momentum = config.optimizer['momentum']
    weight_decay = config.optimizer['weight_decay']
    if torch.cuda.device_count() < 2:
        config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if config.device == 1:  # only for 2 GPUs max
            config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_data_folder = config.val_data_root
    input_size = (int(config.model['input_size']), int(config.model['input_size']))

    # Custom dataloaders
    if config.data_name.upper() == 'COCO':
        test_dataset = COCO17Dataset(val_data_folder, split='val', input_size=input_size, config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.internal_batchsize, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    elif config.data_name.upper() == 'VOC':
        test_dataset = PascalVOCDataset(val_data_folder, split='val', input_size=input_size, config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.internal_batchsize, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    else:
        raise NotImplementedError

    assert args.load_path is not None
    checkpoint = torch.load(args.load_path)
    model = checkpoint['model']
    saved_epoch = checkpoint['epoch']
    model = model.to(config.device)
    optimizer = checkpoint['optimizer']
    print('Evaluate model from checkpoint %s at epoch %d.\n' % (args.load_path, saved_epoch))

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_H-%M-%S")
    config.logger = create_logger('global_logger', os.path.join(config.log_path,
                                                                'eval_result_{}_{}.txt'.format(config.model['arch'],
                                                                                               date_time)))
    print('Length of Testing Dataset:', len(test_dataset))
    print('evaluate checkpoint: ', args.load_path, ' at epoch: ', saved_epoch)
    evaluate(test_loader, model, optimizer, config=config)


def evaluate(test_loader, model, optimizer, config):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.train()

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
            images = images.to(config.device)
            boxes = [b.to(config.device) for b in boxes]
            labels = [l.to(config.device) for l in labels]
            difficulties = [d.to(config.device) for d in difficulties]

            # Forward prop.
            time_start = time.time()
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = \
                        detect(predicted_locs,
                               predicted_scores,
                               min_score=config.nms['min_score'],
                               max_overlap=config.nms['max_overlap'],
                               top_k=config.nms['top_k'], priors_cxcy=model.priors_cxcy,
                               device=model.device)

            time_end = time.time()
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200
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
    details = pprint.pformat(APs)
    config.logger.info(details)

    str_print = 'EVAL: Mean Average Precision {0:.3f}, avg speed {1:.2f} Hz'.format(mAP, 1. / np.mean(detect_speed))
    config.logger.info(str_print)

    del predicted_locs, predicted_scores, boxes, labels

    return APs, mAP


if __name__ == '__main__':
    main()
