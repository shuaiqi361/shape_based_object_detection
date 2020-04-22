import json
import os
import xml.etree.ElementTree as ET
import random

# traffic_labels = ['car', 'pickup', 'truck', 'van', 'bus']
traffic_labels = ['car', 'van', 'bus']
traffic_label_map = {k: v + 1 for v, k in enumerate(traffic_labels)}
traffic_label_map['background'] = 0
rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}  # Inverse mapping

'''
There are 82085 training images containing a total of 594555 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC.
There are 56167 validation images containing a total of 658859 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC.
'''


def parse_annotation(annotation_path, image_folder, down_sample=False):
    tree = ET.parse(annotation_path)

    root = tree.getroot()
    object_list = list()
    image_paths = list()

    for object in root.iter('frame'):
        if random.random() > 0.2 and down_sample:
            continue
        boxes = list()
        labels = list()
        difficulties = list()

        frame_id = int(object.attrib['num'])
        image_name = 'img{:05d}.jpg'.format(frame_id)

        target_list = object.find('target_list')
        for target in target_list:
            uni_name = target.find('attribute').attrib['vehicle_type']
            if uni_name not in traffic_labels:
                continue
            else:
                uni_label = traffic_label_map[uni_name]

            bbox = target.find('box').attrib
            left = float(bbox['left'])
            top = float(bbox['top'])
            width = float(bbox['width'])
            height = float(bbox['height'])
            xmin = max(int(left), 0)
            ymin = max(int(top), 0)
            xmax = int(left + width) - 1
            ymax = int(top + height) - 1
            if xmin + 2 > xmax or ymin + 2 > ymax:
                print('Invalid bbox: ', os.path.join(image_folder, image_name))
                continue

            difficult = 0

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(uni_label)
            difficulties.append(difficult)

        if len(boxes) == 0:
            print('Images with no objects: ', os.path.join(image_folder, image_name))
            continue
        object_list.append({'boxes': boxes, 'labels': labels, 'difficulties': difficulties, 'image_id': image_name})
        image_paths.append(os.path.join(image_folder, image_name))

    assert len(boxes) == len(labels) == len(difficulties)
    assert len(object_list) == len(image_paths)

    return object_list, image_paths


def create_data_lists_detrac(root_path, output_folder):
    # training data
    dataType = 'Train'
    train_images = list()
    train_objects = list()
    n_object = 0

    annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
    annotation_path = os.path.join(root_path, annotation_folder)
    if not os.path.exists(annotation_path):
        print('annotation_path not exist')
        raise FileNotFoundError

    image_folder = 'Insight-MVT_Annotation_{}'.format(dataType)
    image_root = os.path.join(root_path, image_folder)  # under: sequence_name/image_name

    for video in os.listdir(annotation_path):
        if video.endswith('.xml'):
            objects, image_frames_path = parse_annotation(os.path.join(annotation_path, video),
                                                          os.path.join(image_root, video.strip('.xml')))

            if len(objects) == 0:
                continue
            else:
                for obj in objects:
                    n_object += len(obj['boxes'])

            train_objects += objects
            train_images += image_frames_path

    assert len(train_objects) == len(train_images)
    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(traffic_label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_object, os.path.abspath(output_folder)))

    # test data
    dataType = 'Test'
    test_images = list()
    test_objects = list()
    n_object = 0

    annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
    annotation_path = os.path.join(root_path, annotation_folder)
    if not os.path.exists(annotation_path):
        print('annotation_path not exist')
        raise FileNotFoundError

    image_folder = 'Insight-MVT_Annotation_{}'.format(dataType)
    image_root = os.path.join(root_path, image_folder)  # under: sequence_name/image_name

    for video in os.listdir(annotation_path):
        if video.endswith('.xml'):

            objects, image_frames_path = parse_annotation(os.path.join(annotation_path, video),
                                                          os.path.join(image_root, video.strip('.xml')), True)

            if len(objects) == 0:
                continue
            else:
                for obj in objects:
                    n_object += len(obj['boxes'])

            test_objects += objects
            test_images += image_frames_path

    assert len(test_objects) == len(test_images)
    # Save to file
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_object, os.path.abspath(output_folder)))


if __name__ == '__main__':
    # root_path = '/home/keyi/research/data/DETRAC'
    # output_folder = '/home/keyi/research/code/traffic/detection_research_YorkU/dataset/DETRAC'
    # root_path = '/media/keyi/Data/Research/traffic/data/DETRAC'
    # output_folder = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC'
    root_path = '/home/keyi/research/data/DETRAC'
    output_folder = '/home/keyi/research/code/traffic/shape_based_object_detection/data/DETRAC'

    create_data_lists_detrac(root_path, output_folder)
