#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

video_1_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/Dark_attention_001/live_results/DETRAC/MVI_40902.mkv'
video_2_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/Dark_attention_001/live_results/DETRAC/MVI_40902_mask.mkv'

video_out_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/Dark_attention_001/live_results/DETRAC/MVI_40902_mask_cat.mkv'

cap1 = cv2.VideoCapture(video_1_path)


width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 25
cap2 = cv2.VideoCapture(video_2_path)

video_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width * 2, height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:  # some frames may be skipped by cv2 capture, a bug
        img_concat = np.concatenate((frame1, frame2), axis=1)
        text1 = 'left detection  vs.  right attention map'

        # For pm demo
        cv2.putText(img_concat, text1, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 200), 1, cv2.LINE_AA)
        # cv2.imshow("frame1",frame1)
        # cv2.imshow("frame2",frame2)
        # cv2.imshow("concat", img_concat)
        # cv2.waitKey()
        video_out.write(img_concat)
        cv2.waitKey(1)
    else:
        break

cap1.release()
cap2.release()
video_out.release()

# for line in lines:
#     filename=line.strip("\n")
#     img_1=cv2.imread(image_1_path+filename)
#     img_2=cv2.imread(image_2_path+filename)
#     img_3=cv2.imread(image_3_path+filename)
#     img_concat=cv2.hconcat([img_1,img_2,img_3])
#     cv2.imwrite(image_out_path+filename,img_concat)
# cv2.imshow("img",img_concat)
# cv2.waitKey()
