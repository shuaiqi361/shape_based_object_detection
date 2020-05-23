from torchvision import transforms
from PIL import Image
import os
import sys
import cv2
import torch
import json
import time
import numpy as np
from detect_scripts.detect_tools import detect_objects, detect, detect_refine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
resize = transforms.Resize((512, 512))
# resize = transforms.Resize((300, 300))

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def detect_folder(folder_path, model_path, data_set, meta_data_path, output_path, output_file=None):
    # load model
    checkpoint = torch.load(model_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print(model_path)
    print('Loading checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model.device = device
    model = model.to(device)
    model.eval()

    with open(meta_data_path, 'r') as j:
        traffic_label_map = json.load(j)
    rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(traffic_label_map.keys())}

    if output_file is not None:
        f_out = open(output_file, 'w')

    # load video
    if not os.path.exists(folder_path):
        print('video path incorrect.')
        exit()

    width = 1920
    height = 1440
    fps = 60

    video_out = cv2.VideoWriter(os.path.join(output_path, 'GXAB0755_folder_1800.mkv'),
                                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))

    if output_file is not None:
        f_out = open(output_file, 'w')

    speed_list = list()
    frame_list = os.listdir(folder_path)
    n_frames = len(frame_list)
    for frame_id in range(n_frames):
        if frame_id >= 1800:
            break

        frame_name = 'img{:06d}.png'.format(frame_id + 1)
        frame_path = os.path.join(folder_path, frame_name)
        print("Processing frame: ", frame_id + 1, '/', n_frames)
        # frame = cv2.resize(cv2.imread(frame_path), dsize=(width, height))
        frame = cv2.imread(frame_path)
        annotated_image, time_pframe, frame_info = detect_image(frame, model, 0.05, 0.35, 200,
                                                                rev_traffic_label_map, label_color_map)
        speed_list.append(time_pframe)

        video_out.write(annotated_image)
        f_out.write(str(frame_id) + frame_info)

        cv2.imshow('frame detect', annotated_image)
        # print(str(frame_id) + frame_info)
        # cv2.waitKey()
        # exit()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average speed: {} fps.'.format(1. / np.mean(speed_list)))
    print('Saved to:', output_path)
    print('Video configuration: \nresolution:{}x{}, fps:{}'.format(width, height, fps))


def detect_video(video_path, model_path, data_set, meta_data_path, output_path, output_file=None):
    # load model
    checkpoint = torch.load(model_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print(model_path)
    print('Loading checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model.device = device
    model = model.to(device)
    model.eval()

    with open(meta_data_path, 'r') as j:
        traffic_label_map = json.load(j)
    rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(traffic_label_map.keys())}

    if output_file is not None:
        f_out = open(output_file, 'w')

    # load video
    if not os.path.exists(video_path):
        print('video path incorrect.')
        exit()
    cap_video = cv2.VideoCapture(video_path)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_video.get(cv2.CAP_PROP_FPS))

    video_out = cv2.VideoWriter(os.path.join(output_path, video_path.split('/')[-1]),
                                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))

    speed_list = list()
    frame_id = 0
    while True:
        print("Processing frame: ", frame_id)

        ret, frame = cap_video.read()
        if not ret:
            print('Image rendering done. ')
            break

        annotated_image, time_pframe, frame_info = detect_image(frame, model, 0.08, 0.3, 200,
                                                                rev_traffic_label_map, label_color_map)
        speed_list.append(time_pframe)

        video_out.write(annotated_image)
        f_out.write(str(frame_id) + frame_info)
        frame_id += 1
        cv2.imshow('frame detect', annotated_image)
        # print(str(frame_id) + frame_info)
        # cv2.waitKey()
        # exit()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_video.release()
    f_out.close()
    print('Average speed: {} fps.'.format(1. / np.mean(speed_list)))
    print('Saved to:', os.path.join(output_path, video_path.split('/')[-1]))
    print('Video configuration: \nresolution:{}x{}, fps:{}'.format(width, height, fps))


def detect_image(frame, model, min_score, max_overlap, top_k, reverse_label_map, label_color_map):
    # Transform
    image_for_detect = frame.copy()
    img = cv2.cvtColor(image_for_detect, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image = normalize(to_tensor(resize(im_pil)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    start = time.time()
    # predicted_locs, predicted_scores = model(image.unsqueeze(0))
    _, _, _, _, predicted_locs, predicted_scores, pos_idx = model(image.unsqueeze(0))

    # Detect objects in SSD output
    # det_boxes, det_labels, det_scores = detect(predicted_locs, predicted_scores, min_score=min_score,
    #                                            max_overlap=max_overlap, top_k=top_k,
    #                                            priors_cxcy=model.priors_cxcy)

    det_boxes, det_labels, det_scores = detect_refine(predicted_locs, predicted_scores, min_score=min_score,
                                               max_overlap=max_overlap, top_k=top_k,
                                               priors_cxcy=model.priors_cxcy, prior_positives_idx=pos_idx)

    stop = time.time()
    # Move detections to the CPU
    det_boxes_percentage = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [im_pil.width, im_pil.height, im_pil.width, im_pil.height]).unsqueeze(0)
    det_boxes = det_boxes_percentage * original_dims

    # Decode class integer labels
    det_labels_id = [l for l in det_labels[0].to('cpu').tolist()]
    det_labels = [reverse_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    det_labels_scores = [s for s in det_scores[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.']
    # i.e. ['background'] in SSD300.detect_objects() in model.py
    annotated_image = frame.copy()

    if det_labels == ['background']:
        return annotated_image, start - stop, '\n'

    # Annotate
    frame_info = ''
    for i in range(len(det_labels)):
        # Boxes
        box_location = det_boxes[i].tolist()
        box_coordinates = det_boxes_percentage[i].tolist()
        cv2.rectangle(annotated_image, pt1=(int(box_location[0]), int(box_location[1])),
                      pt2=(int(box_location[2]), int(box_location[3])),
                      color=hex_to_rgb(label_color_map[det_labels[i]]), thickness=2)

        # Text
        text = det_labels[i].upper()
        label_id = str(det_labels_id[i])
        label_score = det_labels_scores[i]
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
        text_location = [box_location[0] + 1, box_location[1] + 1, box_location[0] + 1 + label_size[0][0],
                         box_location[1] + 1 + label_size[0][1]]
        cv2.rectangle(annotated_image, pt1=(int(text_location[0]), int(text_location[1])),
                      pt2=(int(text_location[2]), int(text_location[3])),
                      color=(128, 128, 128), thickness=-1)
        cv2.putText(annotated_image, text, org=(int(text_location[0]), int(text_location[3])),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.4, color=(255, 255, 255))

        per_object_prediction_info = ' {0} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.3f}'.format(label_id,
                                                                                      box_coordinates[0],
                                                                                      box_coordinates[1],
                                                                                      box_coordinates[2],
                                                                                      box_coordinates[3],
                                                                                      label_score)
        frame_info += per_object_prediction_info

    return annotated_image, - start + stop, frame_info + '\n'


def print_help():
    print('Try one of the following options:')
    print('python detect_bbox --folder(detect for all images under the folder)')
    print('python detect_bbox --video(detect for all frames in the video)')
    print('python detect_bbox --image(detect for a single image)')
    print('saved images will be put in the same location as input with some suffix')
    exit()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_help()
        exit()
    if sys.argv[1] == '--video':
        # root_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment'
        # video_path = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_demo4.mkv'
        # model_path = os.path.join(root_path, 'SSD512_traffic_002/snapshots/checkpoint_epoch-20.pth.tar')
        # data_set = 'traffic'
        # meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/DETRAC/label_map.json'
        # output_path = os.path.join(root_path, 'SSD512_traffic_002/live_results')
        # detect_video(video_path, model_path, data_set, meta_data_path, output_path)

        # root_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/SSD512_exp_003'
        # video_path = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_demo4.mkv'
        # model_path = os.path.join(root_path, 'snapshots/ssd512_traffic_checkpoint_epoch-3.pth.tar')
        # data_set = 'traffic'
        # meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/DETRAC/label_map.json'
        # output_path = os.path.join(root_path, 'live_results')
        # output_file = os.path.join(output_path, video_path.split('/')[-1].strip('.mkv') + '.txt')
        # detect_video(video_path, model_path, data_set, meta_data_path, output_path, output_file)

        root_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/SSD512_exp_003'
        video_path = '/media/keyi/Data/Research/traffic/data/Transplan/GXAB0755_demo2.MP4'
        model_path = os.path.join(root_path, 'snapshots/ssd512_traffic_checkpoint_epoch-3.pth.tar')
        data_set = 'traffic'
        meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/DETRAC/label_map.json'
        output_path = os.path.join(root_path, 'live_results')
        output_file = os.path.join(output_path, video_path.split('/')[-1].strip('.MP4') + '.txt')
        detect_video(video_path, model_path, data_set, meta_data_path, output_path, output_file)
    elif sys.argv[1] == '--folder':
        # root_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment'
        # folder_path = '/media/keyi/Data/Research/traffic/synthetic_data/GTA_itstraffic/object/image_2'
        # model_path = os.path.join(root_path, 'SSD512_traffic_002/snapshots/checkpoint_epoch-20.pth.tar')
        # data_set = 'traffic'
        # meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/DETRAC/label_map.json'
        # output_path = os.path.join(root_path, 'SSD512_traffic_002/live_results/GTA')
        # detect_folder(folder_path, model_path, data_set, meta_data_path, output_path)

        # root_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/SSD512_exp_003'
        # folder_path = '/media/keyi/Data/Research/traffic/data/Transplan/frames'
        # model_path = os.path.join(root_path, 'snapshots/ssd512_traffic_checkpoint_epoch-3.pth.tar')
        root_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/RefineDet_exp_001'
        folder_path = '/media/keyi/Data/Research/traffic/data/Transplan/frames'
        model_path = os.path.join(root_path, 'snapshots/refinedet_voc_checkpoint_epoch-192.pth.tar')
        data_set = 'traffic'
        meta_data_path = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/label_map.json'
        # meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/DETRAC/label_map.json'
        output_path = os.path.join(root_path, 'live_results')
        output_file = os.path.join(output_path, 'GXAB0755_folder_VOC.txt')
        detect_folder(folder_path, model_path, data_set, meta_data_path, output_path, output_file)

        # root_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/experiment'
        # folder_path = '/media/keyi/Data/Research/traffic/synthetic_data/GTA_itstraffic/object/image_2'
        # model_path = os.path.join(root_path, 'SSD300_traffic_001/snapshots/checkpoint_epoch-60.pth.tar')
        # data_set = 'traffic'
        # meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/Citycam/label_map.json'
        # output_path = os.path.join(root_path, 'SSD300_traffic_001/live_results/GTA')
        # detect_folder(folder_path, model_path, data_set, meta_data_path, output_path)
