import pandas as pd

raw_path = r'C:\Users\hw\Desktop\舌象4.4-4.5.xlsx'
outpath = r"tongue.xlsx"


def tongue_excel_preprocess(raw_path, outpath):
    """
        这个函数处理Excel表格，并将处理过的数据写入新的Excel文件。
        This function processes Excel spreadsheets and writes the processed data to a new Excel file.

        参数(Args):
            raw_path (str): 原始Excel文件的路径。The path of the original Excel file.
            outpath (str): 输出Excel文件的路径。The path of the output Excel file.

        这个函数会执行以下操作:
        This function performs the following operations:
            1. 从raw_path读取Excel文件，加载为一个Pandas DataFrame对象。Read the Excel file from raw_path and load it as a Pandas DataFrame object.
            2. 对舌色、苔色、苔薄厚、苔腻否、证型等特征进行数字映射。Map the features such as tongue color, coating color,
            coating thickness, coating greasy or not, and syndrome type to numbers.
            3. 计算并打印舌腻否特征值的数量。Calculate and print the number of feature values of whether the coating is
            greasy or not.
            4. 将NaN值替换为预设值。Replace NaN values with preset values.
            5. 将处理过的数据写入新的Excel文件，保存在outpath指定的路径。Write the processed data to
            a new Excel file and save it in the path specified by outpath.

        返回(Returns):
            无。None.
    """
    df = pd.read_excel(raw_path)

    s_label = {'淡白': '0', '青紫': '1', '绛': '2', '红': '3', '淡红': '4'}
    df['舌色'] = df['舌色'].map(s_label)

    t_label = {'灰': '0', '白': '1', '黄': '2'}
    df['苔色'] = df['苔色'].map(t_label)

    h_label = {'厚': '0', '薄': '1'}
    df['苔薄厚'] = df['苔薄厚'].map(h_label)

    n_label = {'腻': '0', '否': '1'}
    df['苔腻否'] = df['苔腻否'].map(n_label)

    a = b = c = d = e = 0
    for i in range(len(df['苔腻否'])):
        if df['苔腻否'][i] == '0':
            a += 1
        elif df['苔腻否'][i] == '1':
            b += 1

    print(a, b, c, d, e)
    z_label = {'肝郁脾虚': '1', '肾虚肝郁': '2'}
    df['证型'] = df['证型'].map(z_label)
    df['证型'] = df['证型'].fillna('0')

    df.to_excel(outpath, index=None)


if __name__ == '__main__':
    tongue_excel_preprocess(raw_path, outpath)
