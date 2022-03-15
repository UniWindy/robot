import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from skimage.measure import compare_ssim
import itertools
from PIL import Image, ImageDraw
from numpy.linalg import inv
import imutils
# import zivid
import argparse
import os
import glob


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M


def detect(temp, test, num):
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    w = temp.shape[1]
    h = temp.shape[0]
    res = []
    for i in range(num * num):
        x1 = int(i % num * w / num)
        x2 = int((i % num + 1) * w / num)
        y1 = int(i // num * h / num)
        y2 = int((i // num + 1) * h / num)
        img2 = test[y1:y2, x1:x2]
        img1 = temp[y1:y2, x1:x2]
        ssim = compare_ssim(img1, img2, gaussian_weights=True)
        res.append(ssim)
    return res


def function_good_match(des1, des2, delta=0.75):
    bfm = cv2.BFMatcher()
    matches = bfm.knnMatch(des1, des2, k=2)
    good_match = []
    for m1, m2 in matches:
        if m1.distance < delta * m2.distance:
            good_match.append(m1)
    return good_match


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='./test_images/', help='path to test images')
    parser.add_argument('--abnormal_path', default='./abnormal/', help='path to saved abnormal images')
    parser.add_argument('--normal_path', default='./normal/', help='path to saved normal images')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    for img in os.listdir(args.normal_path):
        os.remove(args.normal_path + img)
    for img in os.listdir(args.abnormal_path):
        os.remove(args.abnormal_path + img)

    sift = cv2.SIFT_create()
    tmp = cv2.imread('./tmp.jpg')
    kp_tmp, des_tmp = sift.detectAndCompute(tmp, None)
    tree = ET.parse('./tmp.xml')
    root = tree.getroot()
    for object in root.findall('object'):
        bnd = object.find('bndbox')
        xmin = int(bnd.find('xmin').text)
        ymin = int(bnd.find('ymin').text)
        xmax = int(bnd.find('xmax').text)
        ymax = int(bnd.find('ymax').text)

    tmp2 = tmp[ymin:ymax, xmin:xmax]
    pts = [[33, 43], [939, 60], [947, 607], [25, 626]]
    tmp2, M = four_point_transform(tmp2, np.array(pts))

    test_path = glob.glob(os.path.join(args.test_path, '*.jpg'))
    for img_path in test_path:
        dst = cv2.imread(img_path)
        dst_copy = dst.copy()
        kp_dst, des_dst = sift.detectAndCompute(dst, None)

        goodMatch = function_good_match(des_dst, des_tmp)
        if len(goodMatch) > 4:
            ptsA = np.float32([kp_dst[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp_tmp[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4.5
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            imgOut = cv2.warpPerspective(dst, H, (tmp.shape[1], tmp.shape[0]), flags=cv2.INTER_LINEAR)

        dst = imgOut[ymin:ymax, xmin:xmax]
        dst, M = four_point_transform(dst, np.array(pts))

        num = 20
        threshold = 0.7
        res = detect(tmp2, dst, num)
        idx = [[i, res[i]] for i in range(len(res)) if res[i] < threshold]

        if min(res) < threshold:
            flag = False
        else:
            flag = True

        length = len(args.test_path)
        if (flag):
            cv2.imwrite(args.normal_path + img_path[length:], dst_copy)
        else:
            w = dst.shape[1]
            h = dst.shape[0]
            for i in range(len(idx)):
                x1 = int(int(idx[i][0] % num) * w / num)
                x2 = int(int((idx[i][0] % num + 1)) * w / num)
                y1 = int(int(idx[i][0] / num) * h / num)
                y2 = int(int((idx[i][0] / num + 1)) * h / num)
                # print(idx[i])
                cv2.rectangle(dst, (x1, y1), (x2, y2), thickness=3, color=(0, 0, 255))

            image = cv2.warpPerspective(np.array(dst), inv(M), (xmax - xmin, ymax - ymin))
            image = np.pad(image, ((ymin, dst_copy.shape[0] - ymax), (xmin, dst_copy.shape[1] - xmax), (0, 0)))
            image = cv2.warpPerspective(np.array(image), inv(H), (dst_copy.shape[1], dst_copy.shape[0]))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_not(mask)
            dst_copy = cv2.bitwise_and(dst_copy, dst_copy, mask=mask)
            result = cv2.add(dst_copy, image)
            length = len(args.test_path)
            cv2.imwrite(args.abnormal_path + img_path[length:], result)
