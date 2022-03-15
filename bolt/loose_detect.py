# -*- encoding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools


def retain_red(image):
    # 提取红色区域
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # 红色在hsv上分布有两种情况
    lower_red1 = np.array([0, 30, 46])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 30, 46])
    upper_red2 = np.array([179, 220, 255])
    # lower_red1 = lower_red2
    # upper_red1 = upper_red2
    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    res1 = cv2.bitwise_and(image, image, mask=mask1)
    res2 = cv2.bitwise_and(image, image, mask=mask2)
    res = cv2.bitwise_or(res1, res2)
    return res


def remove_noisy(img, size):
    # 闭运算平滑
    img_filtered = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    close = cv2.morphologyEx(img_filtered, cv2.MORPH_ERODE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size + 2, size + 2))
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernel)
    return close


def min_rectangle(img, threshold):
    # 绘出最小外接矩形
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    img_rect = img.copy()
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < threshold:
            rect = cv2.minAreaRect(contours[i])  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            cv2.drawContours(img_rect, [box], 0, (0, 0, 0), -1)  # 在图型上绘制矩形
        else:
            rect = cv2.minAreaRect(contours[i])  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            cv2.drawContours(img_rect, [box], 0, (255, 255, 255), 2)  # 在图型上绘制矩形
            box = box.tolist()
            box.append(np.int0(rect[0]).tolist())  #
            box.append(-rect[2])
            rects.append(box)
        for i in range(len(rects)):
            cv2.putText(img_rect, str(round(rects[i][5], 1)), tuple(rects[i][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img_rect, rects


def detect(img, img_rect, rects, threshold):
    if (len(rects) == 0):
        return img, 0
    img_result = img.copy()
    flag = 0
    x = []
    y = []
    for i in range(len(rects)):
        for j in range(4):
            x.append(rects[i][j][0])
            y.append(rects[i][j][1])
    for i, j in itertools.product(range(len(rects)), range(len(rects))):
        # if isOne[j, k] == 1.0 and abs(
        #         bolts[i][j][5] - bolts[i][k][5]) > threshold1 and abs(
        #         bolts[i][j][5] - bolts[i][k][5] - 180) > threshold1 and abs(
        #         bolts[i][j][5] - bolts[i][k][5] + 180) > threshold1:
        if abs(rects[i][5] - rects[j][5]) > threshold and \
                abs(rects[i][5] - rects[j][5] - 90) > threshold and \
                abs(rects[i][5] - rects[j][5] + 90) > threshold:
            flag = 1
            break
    xmax = max(x)
    xmin = min(x)
    ymax = max(y)
    ymin = min(y)
    box = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    if flag:
        cv2.drawContours(img_result, [np.array(box)], 0, (255, 0, 0), 2)
    else:
        cv2.drawContours(img_result, [np.array(box)], 0, (0, 255, 0), 2)

    return img_result, flag


def cal(img, cls):
    W = img.shape[1]
    H = img.shape[0]
    threshold1 = W * H / 200
    threshold2 = 40
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_red = retain_red(img)
    img_filtered = remove_noisy(img_red, 3)
    img_rect, rects = min_rectangle(img_filtered, threshold1)
    img_result, flag = detect(img, img_rect, rects, threshold2)
    return flag


if __name__ == '__main__':
    time_start = time.time()
    img = cv2.imread('1.jpg')
    W = img.shape[1]
    H = img.shape[0]
    threshold1 = W * H / 200
    threshold2 = 40
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_red = retain_red(img)
    img_filtered = remove_noisy(img_red, 3)
    img_rect, rects = min_rectangle(img_filtered, threshold1)
    img_result, flag = detect(img, img_rect, rects, threshold2)
    time_end = time.time()
    plt.figure(dpi=100, figsize=(4, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(img_filtered)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(img_rect)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow(img_result)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("计算时间:", time_end - time_start, "s")
