import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import scipy.io as sio

# 全局数据: X->脑电信号(EEG) shape as (sample_num,seqlen,eeg_channel,bands)
# y->抑郁标签  shape as (sample_num,label)
# pics->舌部图像 shape as (sample_num,rgb_channel,h,w)

X = np.empty([0, 74, 16, 5])
y = np.empty([0, 1])
pics = np.empty([0, 3, 224, 224])


def rotate_image(img):
    """
    这个函数对输入的图像进行随机旋转，作为图像增强的一种方式。
    This function randomly rotates the input image as a form of image augmentation.

    参数(Args):
        img (Image): 待旋转的图像
        img (Image): The image to be rotated.

    返回(Returns):
        Image: 旋转后的图像
        Image: The rotated image.
    """
    angle = np.random.randint(-14, 15)
    img = img.rotate(angle)
    return img


def crop_image(image_path):
    """
       这个函数对输入路径的图像进行裁剪，取中心及周边四个角的224x224大小的图像，并对每个裁剪部分进行旋转增强。
       This function crops the image at the input path, taking the center and
       four corners of the 224x224 size image, and performs rotation augmentation on each cropped part.

       它还将所有图像数据标准化到特定的范围。
       It also normalizes all image data to a specific range.

       参数(Args):
           image_path (str): 图像文件的路径
           image_path (str): Path of the image file.

       返回(Returns):
           np.ndarray: 一系列处理后的图像数据，形状为 (10, 3, 224, 224)
           np.ndarray: A series of processed image data, shape is (10, 3, 224, 224).
    """

    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    width, height = img.size
    crop_size = 224
    center_x, center_y = width // 2, height // 2
    img = img.resize((384, 384), Image.BILINEAR)

    crops = []
    offsets = [(0, 0), (-10, 0), (10, 0), (0, -10), (0, 10)]

    for offset in offsets:
        left_top_x = max(0, center_x - crop_size // 2 + offset[0])
        left_top_y = max(0, center_y - crop_size // 2 + offset[1])
        right_bottom_x = min(width, left_top_x + crop_size)
        right_bottom_y = min(height, left_top_y + crop_size)
        crops.append(img.crop((left_top_x, left_top_y, right_bottom_x, right_bottom_y)))

    for i in range(5):
        crops.append(rotate_image(crops[i]))
    rs = []
    for img in crops:
        img = np.array(img).astype('float32')
        img -= [127.5, 127.5, 127.5]  # 可以不使用这一步,这个等于说后面归一化到[-1,1]而不是[0,1]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img *= 0.007843  # 像素值归一化
        print(img.shape)
        # print(img)
        rs.append(img)
    rs = np.array(rs)
    print(rs.shape)
    return rs


def mat2np(file, filelabel):
    """
       读取MAT文件并转化为Numpy数组，同时生成相应的标签

       Args:
           file (str): MAT文件的路径
           filelabel (int): 与MAT文件相关联的标签

       Returns:
           tuple: 一个元组，其中第一个元素是一个4D的Numpy数组（(sample_num,seqlen,eeg_channel,bands)，
           第二个元素是一个1D的Numpy数组，包含对应的标签
       """
    data = sio.loadmat(file)
    data = data['data']
    data = data[10:750]

    label = np.array([])
    data = data.reshape(-1, 74, 16, 5)
    label = np.append(label, [filelabel] * data.shape[0])

    return data, label


def eeg_preprocess(X, y, pics):
    """
        这个函数预处理图像和数据文件，然后将处理后的结果保存为numpy数组。
        This function preprocesses image and data files, then saves the processed results as numpy arrays.

        参数(Args):
            X (numpy.array): 用于存放数据文件的数组。An array used to store the data files.
            y (numpy.array): 用于存放标签的数组。An array used to store the labels.
            pics (numpy.array): 用于存放图片文件的数组。An array used to store the image files.

        这个函数会执行以下操作:
        This function performs the following operations:
            1. 在给定的文件路径下列出所有的文件。List all files under the given file path.
            2. 为每个文件分配标签。Assign labels to each file.
            3. 读取每个文件和对应的图片，并将其数据存储到X, y, 和pics中。Read each file and corresponding image, and store its data in X, y, and pics.
            4. 保存处理后的数据和图片。Save the processed data and images.

        数据和图片保存在与脚本相同的目录下，以.npy格式（numpy格式）保存。
        The data and images are saved in the same directory as the script, in .npy format (numpy format).

        返回(Returns):
            无。None.
        """
    # 第一步: 列举所有目录下文件名字
    file_path = r'C:\Users\hw\Desktop\eeg_平均值——0.1\\'
    pic_path = r'C:\Users\hw\Desktop\舌_纯舌面0405\\'

    onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]  # 字符串的列表
    print(len(onlyfiles))

    # 第二步: 给每个文件打标 MODMA:抬头0201为0(抑郁),02/03为1(健康) 抑郁 7 21 24 35 37

    filelabels = []
    for i in onlyfiles:
        label = 0 if i[0:2] == '35' or i[0:2] == '37' else 1
        filelabels.append(label)

    # 第三步: 每个文件读取,存放X,y
    ysplits = []
    for i in tqdm(range(len(onlyfiles))):
        file_name = file_path + onlyfiles[i]
        if "VFT" not in onlyfiles[i]:  # 这里是判断任务种类,这里使用了情绪数据(中文拼音qingxu)
            picno = onlyfiles[i].split('_')[0]
            picstr = pic_path + picno + '.png'
            temps = crop_image(picstr)
            decomposed_de, label = mat2np(file_name, filelabels[i])
            pics = np.vstack([pics, temps])
            X = np.vstack([X, decomposed_de])
            y = np.append(y, label)

    # print(pics.shape)
    # print(X.shape)
    # print(y.shape)
    pics_path = r"C:\Users\hw\Desktop\SE_pics_qingxu"
    X_path = r"C:\Users\hw\Desktop\SE_X_16ch_qingxu"
    y_path = r"C:\Users\hw\Desktop\SE_Y_16ch_qingxu"
    np.save(pics_path, pics)
    np.save(X_path, X)
    np.save(y_path, y)


if __name__ == '__main__':
    eeg_preprocess(X, y, pics)
