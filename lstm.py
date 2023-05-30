import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, BatchNormalization
import openpyxl
import datetime as dt
import torch
import matplotlib.pyplot as plt
import os

import emd




# 从文件中加载数据
def load_data(window, filename='data.xlsx', sheetbook='Sheet2', emdsign=False):
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook[sheetbook]

    y = []

    lst = -1
    lstv = 0
    maxy = 0
    for i in range(2, sheet.max_row + 1):
        time_stamp = sheet['A' + str(i)].value
        time_idx = int((time_stamp - dt.datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")).total_seconds()) // 900
        time_stamp = sheet['B' + str(i)].value

        if time_idx <= lst:
            continue

        if time_idx != lst + 1:
            for j in range(lst + 1, time_idx):
                v = (lstv * (j - lst) + time_stamp * (time_idx - j)) / (time_idx - lst)
                y.append(v)

        y.append(time_stamp)
        lst = time_idx
        lstv = time_stamp
        if y[-1] > maxy:
            maxy = y[-1]

    ny = []
    for i in range(len(y)):
        ny.append(y[i] / maxy)

    dy = [0]
    maxdy = 0
    for i in range(1, len(y)):
        dy.append(y[i] - y[i - 1])
        if abs(dy[-1]) > maxdy:
            maxdy = abs(dy[-1])

    ndy = []
    for i in range(len(dy)):
        ndy.append(dy[i] / maxdy)

    # 这样的emd还不是很完美的emd，其实应该在划分数据集之前分解，后续再想办法吧
    if emdsign:
        t, s = emd.define_signal(y)
        fy = emd.remove_IMF1(t, s)
        # emd.draw_new_signal(t, s, y)
    else:
        fy = y

    final_x = torch.DoubleTensor([dy, ndy, y, ny]).transpose(0, 1).reshape(-1, window, 4)
    final_y = torch.DoubleTensor(fy).reshape(-1, window)

    return final_x, final_y, torch.DoubleTensor(fy).reshape(-1, window)




# 进行按window合并
def merge_data(x, fy, ty, window, encoder_len=1):
    # print(x.shape, y.shape)
    xx = []
    for i in range(x.shape[0] - encoder_len):
        xx.append(x[i:i + encoder_len, :, :].reshape(-1, window * encoder_len, 4))
    xx = torch.cat(xx, dim=0)
    fyy = fy[encoder_len:, :]
    tyy = ty[encoder_len:, :]
    return xx, fyy, tyy

'''
Todo:
    a dataloader that can load data automatically rather than split_data
'''




# 按数据集，验证集，测试集划分
def split_data(x, fy, ty, train_len, test_len):
    print(x.shape, fy.shape, ty.shape)
    return x[:train_len, :, :].numpy(), fy[:train_len, :].numpy(), x[train_len:-test_len, :, :].numpy(), ty[train_len:-test_len, :].numpy(), x[-test_len:, :, :].numpy(), ty[-test_len:, :].numpy()




# 训练
def train_lstm(x_train, y_train, x_val, y_val, window, epoch):
    model = Sequential()
    model.add(LSTM(25, dropout=0.1, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(Dense(25))
    model.add(LSTM(25, dropout=0.1))
    model.add(Dense(25))
    model.add(Activation('swish'))
    model.add(Dense(window))
    model.compile(loss='mse', optimizer="adam" )

    for i in range(epoch):
        model.fit(x=x_train, y=y_train, epochs=1, shuffle=False, batch_size=128, validation_data=(x_val, y_val))#参数依次为特征，标签，训练循环次数，小批量（一次放入训练的数据个数）
        model.reset_states()

    return model




# 绘制结果
def plt_pic(yt, yp):
    yyt = []
    yyp = []
    for i in range(len(yt)):
        for j in range(len(yt[i])):
            yyt.append(yt[i][j])

    for i in range(len(yp)):
        for j in range(len(yp[i])):
            yyp.append(yp[i][j])

    mape = 0
    for i in range(len(yyt)):
        mape += abs(yyt[i] - yyp[i]) / abs(yyt[i])
    print(mape / len(yyt))

    x = [i for i in range(len(yyt))]
    plt.plot(x, yyt)
    plt.plot(x, yyp)
    plt.show()





if __name__ == "__main__":
    # 定义基本参数
    window = 1
    encoder_len = 10
    epoch = 100
    train_len = 366 * 96 // window
    test_len = 2 * 96 // window

    x, fy, ty = load_data(window=window, emdsign=False)
    
    x, fy, ty = merge_data(x=x, fy=fy, ty=ty, window=window, encoder_len=encoder_len)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x=x, fy=fy, ty=ty, train_len=train_len, test_len=test_len)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

    model = train_lstm(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, window=window, epoch=epoch)

    y_p = model.predict(x=x_test)
    plt_pic(y_test, y_p)
    