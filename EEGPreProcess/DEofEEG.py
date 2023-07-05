import math
import os

import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

import scipy.io
from mne.preprocessing import compute_proj_ecg
from mne_connectivity import envelope_correlation
import mne


def compute_DE(signal):
    """
    这个函数计算了给定信号的微分熵
    This function computes the Differential Entropy (DE) of a given signal.

    参数(Args):
        signal (array-like): 一维的连续信号数据
        signal (array-like): A one-dimensional continuous signal data.

    返回(Returns):
        float: 输入信号的差分熵值
        float: The Differential Entropy value of the input signal.
    """
    variance = np.var(signal, ddof=1)  # 信号的无偏方差
    return math.log(2 * math.pi * math.e * variance) / 2


raw_eeg_data_path = "F://2023上半年脑电项目//eeg采集-0409版"
de_eeg_data_path = 'C://Users//hw//Desktop//eeg_平均值——0.1//'


def test():
    """
    这个函数从指定路径读取bdf格式的EEG原始数据文件
    This function reads bdf format raw EEG data files from a specified path.

    1)对每个文件的数据进行特定的滤波处理，分别得到delta，theta，alpha，beta，gamma五个频段的信号
    The data of each file is processed with a specific filter to obtain the signals of the delta, theta, alpha, beta, and gamma frequency bands.

    2)计算每个频段的差分熵
    The differential entropy of each frequency band is calculated.

    3)最后将所有频段的差分熵保存为.mat文件，存储在指定的路径中
    Finally, the differential entropies of all frequency bands are saved as a .mat file and stored in a specified path.

    参数(Args): 无
    No arguments needed.

    返回(Returns): 无
    No returns.
    """
    dirs = os.listdir(raw_eeg_data_path)
    for currentFile in dirs:
        fullPath = raw_eeg_data_path + '//' + currentFile
        if currentFile.split('.')[-1] == "bdf":
            fnirs_cw_amplitude_dir = fullPath
            raw = mne.io.read_raw_bdf(fnirs_cw_amplitude_dir)
            raw.load_data()
            epochs = mne.make_fixed_length_epochs(raw, duration=0.1, preload=True)  # 每0.1s作为一个epoch,即de处理的时间区间

            length = len(epochs)  # length为epoch长度 最终.mat数据格式(length,channel,bands)
            tem_data = np.zeros((length, 16, 5))
            delta = epochs.copy().filter(l_freq=1, h_freq=4)
            theta = epochs.copy().filter(l_freq=4, h_freq=8)
            alpha = epochs.copy().filter(l_freq=8, h_freq=14)
            beta = epochs.copy().filter(l_freq=14, h_freq=31)
            gamma = epochs.copy().filter(l_freq=31, h_freq=51)

            for i in range(length):
                for j in range(16):
                    tem_data[i][j][0] = compute_DE(delta.get_data()[i][j])
                    tem_data[i][j][1] = compute_DE(theta.get_data()[i][j])
                    tem_data[i][j][2] = compute_DE(alpha.get_data()[i][j])
                    tem_data[i][j][3] = compute_DE(beta.get_data()[i][j])
                    tem_data[i][j][4] = compute_DE(gamma.get_data()[i][j])
            scipy.io.savemat(de_eeg_data_path + currentFile.split('.')[0] + ".mat", {"data": tem_data})


if __name__ == '__main__':
    test()
