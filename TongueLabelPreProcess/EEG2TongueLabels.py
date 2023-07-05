import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import scipy.io as sio

excel_path = r'tongue.xlsx'
file_path = r'C:\Users\hw\Desktop\eeg_平均值——0.1\\'
other_excel_path = r"C:\Users\hw\Desktop\SE_labels"


def tongue_other_labels_preprocess(excel_path, file_path, other_excel_path):
    """
        这个函数用于为缺少舌类标记的脑电数据补全标记，并将处理后的数据保存为numpy格式。
        This function is used to complete the missing tongue labels for EEG data, and save the processed data in numpy format.

        参数(Args):
            excel_path (str): 包含舌类标记的Excel文件的路径。The path of the Excel file containing the tongue labels.
            file_path (str): 脑电数据文件的路径。The path of the EEG data files.
            other_excel_path (str): 处理后的标记数据保存的路径。The path where the processed label data is saved.

        返回(Returns):
            无。None.
    """

    df = pd.read_excel(excel_path)
    ml = np.empty([0, 4])

    # 第一步: 列举所有目录下文件名字
    onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]  # 字符串的列表
    print(len(onlyfiles))
    # 第二步: 给每个文件打标 MODMA:抬头0201为0(抑郁),02/03为1(健康)

    filelabels = []
    for i in onlyfiles:
        label = 0 if i[0:2] == '35' or i[0:2] == '37' else 1
        filelabels.append(label)

    # 第三步: 每个文件读取,存放X,y

    ysplits = []
    for i in tqdm(range(len(onlyfiles))):
        file_name = file_path + onlyfiles[i]
        if "VFT" in onlyfiles[i]:
            print(onlyfiles[i])
            picno = onlyfiles[i].split('_')[0]
            a = df['舌色'][int(picno) - 1]
            b = df['苔色'][int(picno) - 1]
            c = df['苔薄厚'][int(picno) - 1]
            d = df['苔腻否'][int(picno) - 1]
            label = np.array([a, b, c, d])
            for i in range(10):
                ml = np.vstack([ml, label])
    print(ml.shape)
    np.save(other_excel_path, ml)


if __name__ == '__main__':
    tongue_other_labels_preprocess(excel_path, file_path, other_excel_path)
