import numpy as np

# 这个.py文件起到将脑电通道维度映射到二维脑图的作用,这里以序列长为300,通道数为128的MODMA数据集为例.
# This .py file is used to map the EEG channel dimensions to the 2D brain map,
# here is a MODMA dataset with a sequence length of 300 and 128 channels as an example.
X_path = r"E:\testMODMA\EEGNet_X.npy"
y_path = r"E:\testMODMA\EEGNet_Y.npy"
map_path = r"E:\testMODMA\new_MODMA_CRNN_128.npy"


def map2matrix(X_path, y_path, map_path):
    """
        这个函数将128通道的EEG数据映射到16x19的矩阵上，并将处理过的数据保存为numpy格式。
        This function maps the 128-channel EEG data to a 16x19 matrix, and saves the processed data in numpy format.

        参数(Args):
            X_path (str): 原始EEG数据的路径。The path of the original EEG data.
            y_path (str): 对应原始EEG数据的标签的路径。The path of the labels corresponding to the original EEG data.
            map_path (str): 映射后的EEG数据保存的路径。The path where the mapped EEG data is saved.

        这个函数会执行以下操作:
        This function performs the following operations:
            1. 从X_path和y_path读取EEG数据和对应的标签。Read the EEG data and corresponding labels from X_path and y_path.
            2. 根据预定义的矩阵，将128通道的EEG数据映射到16x19的矩阵上。Map the 128-channel EEG data to a 16x19 matrix according to a predefined matrix.
            3. 将映射后的EEG数据保存在map_path指定的路径，以.npy格式（numpy格式）保存。Save the mapped EEG data to the path specified by map_path, in .npy format (numpy format).

        返回(Returns):
            无。None.
    """

    matrix = [['0', '0', '0', '0', '0', '57', '0', '46', '0', '0', '44', '0', '0', '0', '0', '0', '0', '0', '0'],
              ['0', '0', '0', '0', '51', '50', '47', '45', '41', '40', '39', '43', '34', '38', '0', '0', '0', '0', '0'],
              ['0', '0', '0', '58', '0', '52', '0', '42', '0', '49', '35', '48', '0', '28', '27', '32', '0', '0', '0'],
              ['0', '0', '65', '59', '64', '63', '0', '56', '0', '36', '0', '0', '29', '33', '128', '0', '26', '0',
               '0'],
              ['0', '66', '0', '60', '0', '53', '0', '37', '0', '0', '30', '0', '0', '0', '20', '24', '127', '23', '0'],
              ['70', '69', '0', '68', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '19', '25', '22'],
              ['0', '74', '67', '0', '61', '0', '54', '0', '31', '0', '0', '0', '13', '0', '0', '0', '0', '0', '18'],
              ['0', '71', '73', '0', '0', '0', '0', '0', '0', '0', '0', '7', '0', '0', '0', '12', '0', '21', '16'],
              ['75', '72', '62', '0', '0', '0', '0', '55', '0', '0', '0', '0', '0', '6', '0', '0', '0', '11', '15'],
              ['0', '76', '81', '0', '0', '0', '0', '0', '0', '0', '0', '106', '0', '0', '0', '5', '0', '17', '14'],
              ['0', '82', '77', '0', '78', '0', '79', '0', '80', '0', '0', '0', '112', '0', '0', '0', '0', '0', '10'],
              ['83', '0', '88', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '4', '0', '9'],
              ['0', '84', '0', '85', '94', '86', '0', '87', '0', '0', '105', '0', '0', '0', '118', '124', '126', '3',
               '0'],
              ['0', '89', '90', '91', '0', '99', '0', '0', '0', '104', '0', '0', '111', '0', '123', '0', '2', '8', '0'],
              ['0', '0', '0', '95', '0', '92', '0', '93', '107', '109', '110', '113', '119', '117', '122', '1', '125',
               '0',
               '0'],
              ['0', '0', '0', '96', '97', '100', '98', '101', '102', '103', '108', '114', '115', '116', '120', '121',
               '0',
               '0', '0']]

    matrix = np.array(matrix)
    print(matrix.shape)
    dict = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if int(matrix[i][j]) != 0:
                dict[int(matrix[i][j]) - 1] = (i, j)

    # print(dict)

    X = np.load(X_path)
    y = np.load(y_path)

    # print(X.shape)

    X1619 = np.zeros((len(y), 300, 16, 19, 5))

    for key, value in dict.items():
        X1619[:, :, value[0], value[1], :] = X[:, :, key, :]
    np.save(map_path, X1619)


if __name__ == '__main__':
    map2matrix(X_path, y_path, map_path)
