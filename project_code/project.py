import os
import cv2
from tracker import CentroidTracker
import numpy as np
from collections import defaultdict
from pathlib import Path
import math
from matplotlib import pyplot as plt
from skimage import filters
import random as rng
import sys

"""
# better use json, but don't have enough time
result_dict: 
{frame_id:[detail dict, number of cells, number of mitosis]}
result_dict_detail:
{centroid_key: [(x,y), current speed, total distance, net distance, confinement ratio]}
"""

result_dict = defaultdict(list)
total_cells = 0

def subtraction(img, mask):
    img_out = np.zeros(img.shape, np.uint8)
    (height, width) = img_out.shape
    img_out = img_out.astype(np.float64)
    img = img.astype(np.float64)
    mask = mask.astype(np.float64)
    for x in range(0, height):
        for y in range(0, width):
            img_out[x, y] = img[x, y] - mask[x, y]
    c = np.min(img_out)
    d = np.max(img_out)
    m = 255 / (d - c)
    img_out = np.multiply(np.subtract(img_out, c), m)
    img_out = img_out.astype(np.uint8)
    return img_out


def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def read_imgs(dataset, sequence):
    frames_list = []
    current_path = f'datasets/{dataset}/Sequence {sequence}'
    for f in os.listdir(current_path):
        if os.path.isfile(os.path.join(current_path, f)):
            if f.endswith('.tif'):
                temp_img = cv2.imread(f'{current_path}/{f}', 0)
                frames_list.append(temp_img)
    return frames_list


def process(dataset, sequences, frames):
    i = 0
    ct = CentroidTracker()

    total_distance_dict = {}
    for img in frames:
        result_dict_detail = defaultdict(list)

        original_img = []
        rect_list = []

        if dataset == 'DIC-C2DH-HeLa-DP':
            original_img = read_imgs('DIC-C2DH-HeLa', sequences)
            ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cid, c in enumerate(contours):
                rect = cv2.boundingRect(c)
                if 20 * 20 < cv2.contourArea(c):
                    rect_list.append(rect)
        elif dataset == 'Fluo-N2DL-HeLa':
            black_mask = np.zeros(img.shape, np.uint8)
            sub_black = subtraction(img, black_mask)
            cl = clahe(sub_black)
            ret, thresh = cv2.threshold(cl, 37, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cid, c in enumerate(contours):
                rect = cv2.boundingRect(c)
                if 8 * 8 < cv2.contourArea(c):
                    rect_list.append(rect)
        else:  # PhC-C2DL-PSC
            cl = clahe(img)
            ret, thresh = cv2.threshold(cl, 170, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cid, c in enumerate(contours):
                rect = cv2.boundingRect(c)
                if 3 * 3 < cv2.contourArea(c):
                    rect_list.append(rect)


        if dataset == 'DIC-C2DH-HeLa-DP':
            img_bgr = [0] * 3 * len(original_img[i]) * len(original_img[i][0])
            img_bgr = np.array(img_bgr).reshape(len(original_img[i]), len(original_img[i][0]), 3)
            for xx in range(0, len(original_img[i])):
                for yy in range(0, len(img[0])):
                    img_bgr[xx][yy][0] = original_img[i][xx][yy]
                    img_bgr[xx][yy][1] = original_img[i][xx][yy]
                    img_bgr[xx][yy][2] = original_img[i][xx][yy]
        else:
            img_bgr = [0] * 3 * len(img) * len(img[0])
            img_bgr = np.array(img_bgr).reshape(len(img), len(img[0]), 3)
            for xx in range(0, len(img)):
                for yy in range(0, len(img[0])):
                    img_bgr[xx][yy][0] = img[xx][yy]
                    img_bgr[xx][yy][1] = img[xx][yy]
                    img_bgr[xx][yy][2] = img[xx][yy]

        objects, obj_history = ct.update(rect_list)
        # mitosis detection
        key_list = []
        # pre_key_list = []
        new_cell_num = 0
        keyy = 0
        count = len(objects)
        while count > 0:
            if keyy in objects:
                count -= 1
                key_list.append(keyy)
            keyy += 1
        if i == 0:
            pre_key_list = key_list
            # pre_length = len(objects)
            # new_length = len(objects)

        else:
            if key_list[-1] > pre_key_list[-1]:
                new_cell_num = 1
                while key_list[-1 * new_cell_num] > pre_key_list[-1]:
                    new_cell_num += 1
            pre_key_list = key_list

        if new_cell_num != 0:
            for index in range(0, len(rect_list)):
                x, y, w, h = rect_list[index]
                img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 0), 2)
            for index in range(1, new_cell_num):
                true_key = key_list[-1 * index]
                xx = objects[true_key][0]
                yy = objects[true_key][1]
                for retriv in range(0, len(rect_list)):
                    x, y, w, h = rect_list[retriv]
                    if w % 2 != 0:
                        xxx = x + (w - 1) / 2
                    else:
                        xxx = x + w / 2
                    if h % 2 != 0:
                        yyy = y + (h - 1) / 2
                    else:
                        yyy = y + h / 2
                    if xxx == xx and yyy == yy:
                        img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
                        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        p = 0
                        first_choice = None
                        for second in range(0, len(rect_list)):
                            xxxx, yyyy, wwww, hhhh = rect_list[second]
                            if (abs(xxxx-x)<20 and abs(yyyy-y)<20) and (xxxx!=x and yyyy!=y):
                                if abs(wwww-w)/w<1 and abs(hhhh-h)/h<1:
                                    p += 1
                                    if p == 1:
                                        first_choice = [xxxx, yyyy, wwww, hhhh]
                                    else:
                                        if (x - xxxx) * (x - xxxx) + (y - yyyy) * (y - yyyy) < (
                                                x - first_choice[0]) * (x - first_choice[0]) + (
                                                y - first_choice[1]) * (y - first_choice[1]):
                                            first_choice = [xxxx, yyyy, wwww, hhhh]
                        if first_choice:
                            img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
                            cv2.rectangle(img_bgr, (first_choice[0], first_choice[1]),
                                          (first_choice[0] + first_choice[2], first_choice[1] + first_choice[3]),
                                          (0, 255, 0), 2)
        else:
            for index in range(0, len(rect_list)):
                x, y, w, h = rect_list[index]
                img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 0), 2)

        for objectID, centroid in objects.items():
            # centroids_so_far = []
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            if dataset != 'PhC-C2DL-PSC':
                cv2.putText(img_bgr, str(objectID), (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img_bgr, (centroid[0], centroid[1]), 1, (0, 255, 0), 2)
            result_dict_detail[objectID].append((centroid[0], centroid[1]))

        for key, value in obj_history.items():
            if len(value) > 1:
                for point_i in range(0, len(value) - 1):
                    temp_point = (value[point_i][0], value[point_i][1])
                    temp_next_point = (value[point_i + 1][0], value[point_i + 1][1])
                    cv2.line(img_bgr, temp_point, temp_next_point, (0, 255, 0), 1)
                # first_point_t = (value[0][0], value[0][1])
                # print(f'first_point_t: {first_point_t}')

                # Calculate the Speed of Cell
                current_point = value[-1]
                previous_point = value[-2]
                current_speed = np.linalg.norm(current_point - previous_point)

                # Calculate the Total Distance
                if key in total_distance_dict:
                    total_distance_dict[key] += current_speed
                else:
                    total_distance_dict[key] = current_speed
                total_distance = total_distance_dict[key]

                # Net distance travelled up to that time point
                first_point = value[0]
                net_distance = np.linalg.norm(current_point - first_point)
                # Calculate the Confinement reatio of the cell motion
                if net_distance != 0:
                    confinement_ratio = total_distance / net_distance
                else:
                    confinement_ratio = 0
                result_dict_detail[key].append(round(current_speed, 2))
                result_dict_detail[key].append(round(total_distance, 2))
                result_dict_detail[key].append(round(net_distance, 2))
                result_dict_detail[key].append(round(confinement_ratio, 2))

            elif len(value) == 1:
                result_dict_detail[key].append(0)
                result_dict_detail[key].append(0)
                result_dict_detail[key].append(0)
                result_dict_detail[key].append(0)

        result_dict[i].append(result_dict_detail)
        # print the real time cell number
        (x, y) = (20, 20)
        message = f'The number of cells: {len(rect_list)}'
        global total_cells
        total_cells += len(rect_list)
        cv2.putText(img_bgr, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # print count of the cells that are dividing
        text = f'The number of the cells that are dividing: {new_cell_num}'
        cv2.putText(img_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        Path(f'datasets/{dataset}_{sequences}_out').mkdir(parents=False, exist_ok=True)
        cv2.imwrite(f'datasets/{dataset}_{sequences}_out/{i}.tif', img_bgr)
        result_dict[i].append(len(rect_list))
        result_dict[i].append(new_cell_num)

        i = i + 1


def click_event(frames_len, out_path):
    while 1:
        choice = input('Which frame would like to see? Press q to quit: ')
        if choice == 'q' or choice == 'Q':
            print('Bye!')
            sys.exit()
        elif not str.isdigit(choice):
            print(f'not a positive number')
        elif int(choice) < 0 or int(choice) > frames_len - 1:
            print(f'Invalid frame number, accept 0 - {frames_len - 1}')
        else:
            img_frame = cv2.imread(f'{out_path}/{choice}.tif')
            cv2.namedWindow(f'Frame {choice}')
            parm = [choice]
            cv2.setMouseCallback(f'Frame {choice}', mouse_click, parm)
            while 1:
                cv2.imshow(f'Frame {choice}', img_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_id = param[0]
        details = result_dict[int(frame_id)][0]
        for key, value in details.items():
            centroid = value[0]
            rect_offset = 3
            x1, y1, x2, y2 = centroid[0] - rect_offset, centroid[1] - rect_offset, centroid[0] + rect_offset, centroid[
                1] + rect_offset
            if x1 < x < x2 and y1 < y < y2:
                print('+++++++++++++++++++++++++++++++++++')
                print(f'cell ID:{key} is clicked')
                print(f'current speed is {value[1]} pixels/frame')
                print(f'total distance is {value[2]} pixels')
                print(f'net distance is {value[3]} pixels')
                print(f'confinement ratio is {value[4]}')


def main():
    if len(sys.argv) < 3:
        print("usage: project dataset_name sequences_number flag")
        sys.exit()
    else:
        dataset = sys.argv[1]
        sequences = sys.argv[2]

    if dataset not in {'DIC-C2DH-HeLa', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC'}:
        print("only support the following 3 datasets: DIC-C2DH-HeLa, Fluo-N2DL-HeLa, or PhC-C2DL-PSC")
        sys.exit()
    elif sequences not in {'1', '2', '3', '4'}:
        print("only support sequences number 1, 2, 3, or 4")
        sys.exit()
    if dataset == 'DIC-C2DH-HeLa':
        dataset = 'DIC-C2DH-HeLa-DP'
    frames = read_imgs(dataset, sequences)
    print(f'Read {len(frames)} frames from dataset {dataset} sequences {sequences}')
    print('+++++++++++++++++++++++++++++++++++')
    global total_cells

    process(dataset, sequences, frames)
    print(f'total cells: {total_cells}')
    out_path = f'datasets/{dataset}_{sequences}_out'

    click_event(len(frames), out_path)


if __name__ == '__main__':
    main()
