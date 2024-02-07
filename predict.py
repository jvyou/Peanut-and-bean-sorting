# -*- coding: GB2312 -*-
import os
import cv2
import numpy as np
import torch
from PIL import Image
from utils.model import ResNet18
from torchvision import transforms

path = 'data/pic'
image_path = os.listdir(path)

classify = {0: 'baiban', 1: 'bandian', 2: 'famei', 3: 'faya', 4: 'hongpi', 5: 'qipao', 6: 'youwu', 7: 'zhengchang'}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()])

net = ResNet18(8)
net.load_state_dict(torch.load('model_weights/ResNet18.pth'))

min_size = 30
max_size = 400


# 过滤边框
def delet_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


for i in image_path:
    img = cv2.imread(os.path.join(path,i))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转HSV色彩空间

    # 定义背景颜色区间（蓝色区间）
    lower_blue = np.array([100, 100, 8])
    upper_blue = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 创建掩膜（在上述颜色范围内（背景）为白色，不在（花生豆）则为黑色）

    result = cv2.bitwise_and(img, img, mask=mask)  # 根据掩膜提取图像，会将花生豆的部分变为黑色，然后提取出背景部分
    result = result.astype(np.uint8)

    _, binary_image = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)  # 三通道二值化。背景会全为白色，花生豆部分为黑色

    # 到这里我们就得到了经过掩膜过滤的图片，其中白色的为背景，黑色的为花生豆，我们可以看一下
    cv2.namedWindow('HSV_Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV_Result', 2840, 1000)
    cv2.imshow('HSV_Result', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    inverted_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    _, binary_image = cv2.threshold(inverted_image, 1, 255, cv2.THRESH_BINARY)  # 单通道二值化

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)

    delete_list = []
    for i in range(len(contours)):
        # 通过框的周长去过滤边框
        if (cv2.arcLength(contours[i], True) < min_size) or (cv2.arcLength(contours[i], True) > max_size):
            delete_list.append(i)
    contours = delet_contours(contours, delete_list)

    # 遍历每一个框（取出每一个单独的花生豆进行预测）
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        img_pred = img[y:y+h, x:x+w, :]
        img_pred = Image.fromarray(img_pred)  # 将numpy数组转为PIL图像对象
        img_pred = transform(img_pred)  # 调整图像尺寸和转tensor格式
        img_pred = torch.unsqueeze(img_pred, dim=0)  # 升一个维度
        pred = torch.argmax(net(img_pred), dim=1)  # 拿到概率最大的分类
        preds = classify[int(pred)]  # 数字映射为字符串
        cv2.putText(img, preds, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)  # 写类别标签
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 画矩形框

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result',2840,1000)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
