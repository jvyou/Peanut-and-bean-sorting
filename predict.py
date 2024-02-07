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


# ���˱߿�
def delet_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


for i in image_path:
    img = cv2.imread(os.path.join(path,i))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # תHSVɫ�ʿռ�

    # ���屳����ɫ���䣨��ɫ���䣩
    lower_blue = np.array([100, 100, 8])
    upper_blue = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # ������Ĥ����������ɫ��Χ�ڣ�������Ϊ��ɫ�����ڣ�����������Ϊ��ɫ��

    result = cv2.bitwise_and(img, img, mask=mask)  # ������Ĥ��ȡͼ�񣬻Ὣ�������Ĳ��ֱ�Ϊ��ɫ��Ȼ����ȡ����������
    result = result.astype(np.uint8)

    _, binary_image = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)  # ��ͨ����ֵ����������ȫΪ��ɫ������������Ϊ��ɫ

    # ���������Ǿ͵õ��˾�����Ĥ���˵�ͼƬ�����а�ɫ��Ϊ��������ɫ��Ϊ�����������ǿ��Կ�һ��
    cv2.namedWindow('HSV_Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV_Result', 2840, 1000)
    cv2.imshow('HSV_Result', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    inverted_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)  # ת�Ҷ�ͼ
    _, binary_image = cv2.threshold(inverted_image, 1, 255, cv2.THRESH_BINARY)  # ��ͨ����ֵ��

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)

    delete_list = []
    for i in range(len(contours)):
        # ͨ������ܳ�ȥ���˱߿�
        if (cv2.arcLength(contours[i], True) < min_size) or (cv2.arcLength(contours[i], True) > max_size):
            delete_list.append(i)
    contours = delet_contours(contours, delete_list)

    # ����ÿһ����ȡ��ÿһ�������Ļ���������Ԥ�⣩
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        img_pred = img[y:y+h, x:x+w, :]
        img_pred = Image.fromarray(img_pred)  # ��numpy����תΪPILͼ�����
        img_pred = transform(img_pred)  # ����ͼ��ߴ��תtensor��ʽ
        img_pred = torch.unsqueeze(img_pred, dim=0)  # ��һ��ά��
        pred = torch.argmax(net(img_pred), dim=1)  # �õ��������ķ���
        preds = classify[int(pred)]  # ����ӳ��Ϊ�ַ���
        cv2.putText(img, preds, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)  # д����ǩ
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # �����ο�

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result',2840,1000)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
