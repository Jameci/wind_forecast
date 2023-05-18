import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, BatchNormalization
import openpyxl
import datetime as dt
import torch
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

epoch = 100

def load_data(window, filename='data.xlsx', sheetbook='Sheet2'):
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

    return torch.DoubleTensor([dy, ndy, y, ny]).transpose(0, 1).reshape(-1, window, 4), torch.DoubleTensor(y).reshape(-1, window)

def merge_data(x, y, window, encoder_len=1):
    print(x.shape, y.shape)
    xx = []
    for i in range(x.shape[0] - encoder_len):
        xx.append(x[i:i + encoder_len, :, :].reshape(-1, window * encoder_len, 4))
    xx = torch.cat(xx, dim=0)
    yy = y[encoder_len:, :]
    return xx, yy

'''
Todo:
    a dataloader that can load data automatically rather than split_data
'''

def split_data(x, y, train_len, test_len):
    print(x.shape, y.shape)
    return x[:train_len, :, :].numpy(), y[:train_len, :].numpy(), x[train_len:-test_len, :, :].numpy(), y[train_len:-test_len, :].numpy(), x[-test_len:, :, :].numpy(), y[-test_len:, :].numpy()

def train_lstm(x_train, y_train, x_val, y_val, window):
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
    window = 96
    encoder_len = 10

    train_len = 366 * 96 // window
    test_len = 2 * 96 // window

    x, y = load_data(window=window)
    x, y = merge_data(x=x, y=y, window=window, encoder_len=encoder_len)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x=x, y=y, train_len=train_len, test_len=test_len)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

    model = train_lstm(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, window=window)

    y_p = model.predict(x=x_test)
    plt_pic(y_p, y_test)
