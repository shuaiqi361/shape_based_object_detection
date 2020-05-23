import os

width = 1920.
height = 1440.

result_file = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/SSD512_exp_003/live_results/GXAB0755_folder_1800.txt'
split_result = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/SSD512_exp_003/live_results/GXAB0755_folder_1800_parse.txt'


f_result = open(result_file, 'r')
f_out = open(split_result, 'w')

lines = f_result.readlines()
counter = 0
for line in lines:
    elements = line.strip('\n').split(' ')
    frame_id = elements[0]
    counter += 1
    for i in range((len(elements) - 1) // 6):
        x1 = float(elements[1 + 6 * i + 1]) * width
        y1 = float(elements[1 + 6 * i + 2]) * height
        x2 = float(elements[1 + 6 * i + 3]) * width
        y2 = float(elements[1 + 6 * i + 4]) * height
        score = float(elements[1 + 6 * i + 5])
        newline = '{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.3f}\n'.format(frame_id, x1, y1, x2, y2, score)
        f_out.write(newline)
        # print(newline)


print(counter, " frames done.")
f_result.close()
f_out.close()